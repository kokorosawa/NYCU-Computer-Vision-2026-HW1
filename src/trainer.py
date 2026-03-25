import csv
import os
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
        self.best_epoch = 0

    def train_step(self):
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for images, labels in tqdm(self.train_dataloader, desc="Training"):
            images = images.to(self.device, non_blocking=self.use_amp)
            labels = labels.to(self.device, non_blocking=self.use_amp)

            self.optimizer.zero_grad()
            with torch.amp.autocast(
                device_type=self.amp_device_type,
                enabled=self.use_amp,
                dtype=torch.float16,
            ):
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
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

            if save_path is not None and (
                self.best_val_acc is None or val_acc > self.best_val_acc
            ):
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                self.save_model(save_path)
                print(
                    f"Saved best model to {save_path} "
                    f"(epoch {self.best_epoch}, val acc {self.best_val_acc:.4%})"
                )

        return {
            "best_val_acc": self.best_val_acc if self.best_val_acc is not None else 0.0,
            "best_epoch": self.best_epoch,
        }

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def test(self, test_dataloader=None, output_path="../output/prediction.csv"):
        self.model.eval()
        dataloader = test_dataloader or self.test_dataloader
        results = []

        with torch.no_grad():
            for images, filenames in tqdm(dataloader, desc="Testing"):
                images = images.to(self.device, non_blocking=self.use_amp)
                with torch.amp.autocast(
                    device_type=self.amp_device_type,
                    enabled=self.use_amp,
                    dtype=torch.float16,
                ):
                    outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1).cpu().tolist()

                for filename, pred in zip(filenames, preds):
                    image_name = os.path.splitext(os.path.basename(filename))[0]
                    results.append((image_name, pred))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["image_name", "pred_label"])
            writer.writerows(results)
