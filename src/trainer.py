import csv
import os
import random
import torch
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        scheduler=None,
        train_dataloader=None,
        val_dataloader=None,
        test_dataloader=None,
        device="cuda",
        class_names=None,
        early_stopping_patience=None,
        early_stopping_min_delta=0.0,
        mixup_alpha=0.0,
        cutmix_alpha=0.0,
        mix_prob=0.0,
        tta_horizontal_flip=False,
        wandb_run=None,
    ):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.use_amp = self.device.startswith("cuda")
        self.amp_device_type = "cuda" if self.use_amp else "cpu"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.best_val_acc = None
        self.best_val_loss = None
        self.best_epoch = 0
        self.class_names = class_names
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mix_prob = mix_prob
        self.epochs_without_improvement = 0
        self.tta_horizontal_flip = tta_horizontal_flip
        self.wandb_run = wandb_run

    def _forward_with_tta(self, images):
        with torch.amp.autocast(
            device_type=self.amp_device_type,
            enabled=self.use_amp,
            dtype=torch.float16,
        ):
            outputs = self.model(images)
            if self.tta_horizontal_flip:
                flip_outputs = self.model(torch.flip(images, dims=(-1,)))
                outputs = (outputs + flip_outputs) / 2.0
        return outputs

    def _sample_lambda(self, alpha):
        if alpha <= 0.0:
            return 1.0
        beta_dist = torch.distributions.Beta(alpha, alpha)
        lam = beta_dist.sample().item()
        return float(lam)

    def _rand_bbox(self, size, lam):
        _, _, height, width = size
        cut_ratio = (1.0 - lam) ** 0.5
        cut_w = int(width * cut_ratio)
        cut_h = int(height * cut_ratio)

        cx = random.randint(0, width - 1)
        cy = random.randint(0, height - 1)

        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, width)
        y2 = min(cy + cut_h // 2, height)
        return x1, y1, x2, y2

    def _mix_batch(self, images, labels):
        if self.mix_prob <= 0.0 or random.random() >= self.mix_prob:
            return images, labels, labels, 1.0

        apply_cutmix = self.cutmix_alpha > 0.0 and (
            self.mixup_alpha <= 0.0 or random.random() < 0.5
        )
        indices = torch.randperm(images.size(0), device=images.device)
        shuffled_images = images[indices]
        shuffled_labels = labels[indices]

        if apply_cutmix:
            lam = self._sample_lambda(self.cutmix_alpha)
            x1, y1, x2, y2 = self._rand_bbox(images.size(), lam)
            mixed_images = images.clone()
            mixed_images[:, :, y1:y2, x1:x2] = shuffled_images[:, :, y1:y2, x1:x2]
            patch_area = max(x2 - x1, 0) * max(y2 - y1, 0)
            lam = 1.0 - patch_area / float(images.size(-1) * images.size(-2))
            return mixed_images, labels, shuffled_labels, lam

        if self.mixup_alpha > 0.0:
            lam = self._sample_lambda(self.mixup_alpha)
            mixed_images = lam * images + (1.0 - lam) * shuffled_images
            return mixed_images, labels, shuffled_labels, lam

        return images, labels, labels, 1.0

    def _mixed_loss(self, outputs, labels_a, labels_b, lam):
        if lam >= 1.0:
            return self.loss_fn(outputs, labels_a)
        return lam * self.loss_fn(outputs, labels_a) + (1.0 - lam) * self.loss_fn(
            outputs, labels_b
        )

    def _is_better(self, val_loss, val_acc):
        if self.best_val_acc is None or self.best_val_loss is None:
            return True

        min_delta = self.early_stopping_min_delta
        if val_acc > self.best_val_acc + min_delta:
            return True
        if abs(val_acc - self.best_val_acc) <= min_delta:
            return val_loss < self.best_val_loss - min_delta
        return False

    def train_step(self):
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for images, labels in tqdm(self.train_dataloader, desc="Training"):
            images = images.to(self.device, non_blocking=self.use_amp)
            labels = labels.to(self.device, non_blocking=self.use_amp)
            images, labels_a, labels_b, lam = self._mix_batch(images, labels)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=self.amp_device_type,
                enabled=self.use_amp,
                dtype=torch.float16,
            ):
                outputs = self.model(images)
                loss = self._mixed_loss(outputs, labels_a, labels_b, lam)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            preds = torch.argmax(outputs, dim=1)
            batch_correct = (
                lam * (preds == labels_a).sum().item()
                + (1.0 - lam) * (preds == labels_b).sum().item()
            )
            total_correct += batch_correct
            total_samples += labels_a.size(0)
            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_dataloader)
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def validate_step(self):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for images, labels in tqdm(self.val_dataloader, desc="Validation"):
                images = images.to(self.device, non_blocking=self.use_amp)
                labels = labels.to(self.device, non_blocking=self.use_amp)
                with torch.amp.autocast(
                    device_type=self.amp_device_type,
                    enabled=self.use_amp,
                    dtype=torch.float16,
                ):
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
        avg_loss = total_loss / len(self.val_dataloader)
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def train(self, epochs, save_path=None):
        for epoch in tqdm(range(epochs), desc="Epochs"):
            train_loss, train_acc = self.train_step()
            val_loss, val_acc = self.validate_step()
            if self.scheduler is not None:
                self.scheduler.step()
            current_lr = (
                self.optimizer.param_groups[0]["lr"]
                if self.optimizer is not None
                else 0.0
            )
            print(
                f"Epoch {epoch+1}/{epochs}, "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4%}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4%}, "
                f"LR: {current_lr:.6f}"
            )

            if self.wandb_run is not None:
                self.wandb_run.log(
                    {
                        "epoch": epoch + 1,
                        "train/loss": train_loss,
                        "train/acc": train_acc,
                        "val/loss": val_loss,
                        "val/acc": val_acc,
                        "train/lr": current_lr,
                    }
                )

            if self._is_better(val_loss, val_acc):
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                self.epochs_without_improvement = 0
                if self.wandb_run is not None:
                    self.wandb_run.summary["best_val_acc"] = self.best_val_acc
                    self.wandb_run.summary["best_val_loss"] = self.best_val_loss
                    self.wandb_run.summary["best_epoch"] = self.best_epoch
                if save_path is not None:
                    self.save_model(save_path)
                    print(
                        f"Saved best model to {save_path} "
                        f"(epoch {self.best_epoch}, val acc {self.best_val_acc:.4%}, "
                        f"val loss {self.best_val_loss:.4f})"
                    )
            else:
                self.epochs_without_improvement += 1

            if (
                self.early_stopping_patience is not None
                and self.epochs_without_improvement >= self.early_stopping_patience
            ):
                print(
                    f"Early stopping at epoch {epoch+1}: "
                    f"no improvement for {self.early_stopping_patience} epochs."
                )
                break

        return {
            "best_val_acc": self.best_val_acc if self.best_val_acc is not None else 0.0,
            "best_val_loss": (
                self.best_val_loss if self.best_val_loss is not None else float("inf")
            ),
            "best_epoch": self.best_epoch,
        }

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def test(self, test_dataloader=None, output_path="../output/prediction.csv"):
        filenames, logits = self.predict_logits(test_dataloader)
        pred_indices = torch.argmax(logits, dim=1).cpu().tolist()
        preds = (
            [int(self.class_names[idx]) for idx in pred_indices]
            if self.class_names is not None
            else pred_indices
        )
        results = []
        for filename, pred in zip(filenames, preds):
            image_name = os.path.splitext(os.path.basename(filename))[0]
            results.append((image_name, pred))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["image_name", "pred_label"])
            writer.writerows(results)
        if self.wandb_run is not None:
            self.wandb_run.summary["prediction_output_path"] = str(output_path)

    def predict_logits(self, dataloader=None):
        self.model.eval()
        dataloader = dataloader or self.test_dataloader
        filenames_all = []
        logits_all = []

        with torch.no_grad():
            for images, filenames in tqdm(dataloader, desc="Testing"):
                images = images.to(self.device, non_blocking=self.use_amp)
                outputs = self._forward_with_tta(images)
                filenames_all.extend(filenames)
                logits_all.append(outputs.float().cpu())

        return filenames_all, torch.cat(logits_all, dim=0)

    def predict_labeled_logits(self, dataloader):
        self.model.eval()
        logits_all = []
        labels_all = []

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Validation Inference"):
                images = images.to(self.device, non_blocking=self.use_amp)
                outputs = self._forward_with_tta(images)
                logits_all.append(outputs.float().cpu())
                labels_all.append(labels.cpu())

        return torch.cat(logits_all, dim=0), torch.cat(labels_all, dim=0)
