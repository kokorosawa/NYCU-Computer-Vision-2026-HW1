import argparse
import csv
import glob
import math
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from dataset import TestDataset
from model import Resnet
from trainer import Trainer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_BATCH_SIZE = 256
DEFAULT_EPOCHS = 20
DEFAULT_IMAGE_SIZE = 256
DEFAULT_MODEL = "resnet50"
MODEL_CHOICES = (
    "resnet18",
    "resnet34",
    "resnet50",
    "resnext50_32x4d",
    "eca_resnet18",
    "eca_resnet34",
    "eca_resnet50",
    "gmlp_resnet50",
    "se_resnet18",
    "se_resnet34",
    "se_resnet50",
)
DEFAULT_NUM_WORKERS = 8
DEFAULT_OPTIMIZER = "adamw"
DEFAULT_SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_wandb_tags(tags):
    if tags is None:
        return None
    return [tag.strip() for tag in tags.split(",") if tag.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Train and test image classifier")
    parser.add_argument(
        "--mode",
        choices=("train", "test", "all", "tune"),
        default="all",
        help="Run training, testing, or both.",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default=DEFAULT_MODEL,
        help="Backbone model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for all dataloaders.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--backbone-lr-scale",
        type=float,
        default=0.1,
        help="Learning rate scale applied to backbone parameters.",
    )
    parser.add_argument(
        "--optimizer",
        choices=("adamw", "sgd"),
        default=DEFAULT_OPTIMIZER,
        help="Optimizer for training.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate for the classifier head.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor for cross entropy.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio before cosine decay.",
    )
    parser.add_argument(
        "--eta-min",
        type=float,
        default=1e-6,
        help="Minimum learning rate for cosine decay.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Stop training when validation stops improving for this many epochs.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum improvement required for validation metrics.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Input image size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze pretrained backbone and train classification head only.",
    )
    parser.add_argument(
        "--randaugment-magnitude",
        type=int,
        default=9,
        help="RandAugment magnitude. Set below 0 to disable.",
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.2,
        help="MixUp beta distribution alpha. Set to 0 to disable.",
    )
    parser.add_argument(
        "--cutmix-alpha",
        type=float,
        default=1.0,
        help="CutMix beta distribution alpha. Set to 0 to disable.",
    )
    parser.add_argument(
        "--mix-prob",
        type=float,
        default=0.5,
        help="Probability of applying MixUp or CutMix to a batch.",
    )
    parser.add_argument(
        "--tta-horizontal-flip",
        action="store_true",
        help="Average test logits with a horizontally flipped copy of each image.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Dataset root containing train/val/test.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Checkpoint path. Defaults to models/<model>.pth.",
    )
    parser.add_argument(
        "--resume-train",
        action="store_true",
        help="Load model weights from model-path before continuing training.",
    )
    parser.add_argument(
        "--ensemble-models",
        nargs="+",
        default=None,
        help="Model names for ensemble inference. Test mode only.",
    )
    parser.add_argument(
        "--ensemble-model-paths",
        nargs="+",
        type=Path,
        default=None,
        help="Checkpoint paths for ensemble inference. Must align with ensemble-models.",
    )
    parser.add_argument(
        "--ensemble-top-k",
        type=int,
        default=None,
        help="Select the top-k checkpoints by validation accuracy before ensemble inference.",
    )
    parser.add_argument(
        "--ensemble-candidate-glob",
        type=str,
        default="models/optuna_trials/*.pth",
        help="Glob used to discover candidate checkpoints when ensemble-top-k is set without explicit ensemble-model-paths.",
    )
    parser.add_argument(
        "--ensemble-split",
        choices=("test", "val"),
        default="test",
        help="Dataset split for ensemble inference.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROJECT_ROOT / "output" / "prediction.csv",
        help="Prediction csv output path.",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=20,
        help="Number of Optuna trials when mode=tune.",
    )
    parser.add_argument(
        "--optuna-timeout",
        type=int,
        default=None,
        help="Optional Optuna timeout in seconds.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use, e.g. cuda or cpu.",
    )
    parser.add_argument(
        "--wandb-project",
        default="dlcv-hw1",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="Weights & Biases entity/team name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional Weights & Biases run name.",
    )
    parser.add_argument(
        "--wandb-tags",
        default=None,
        help="Comma-separated Weights & Biases tags.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_train_transform(image_size):
    transform_list = [
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=12, translate=(0.08, 0.08), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.03),
    ]
    return transforms.Compose(
        transform_list
        + [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
        ]
    )


def build_train_transform_with_randaugment(image_size, randaugment_magnitude):
    transform_list = [
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    if randaugment_magnitude >= 0:
        transform_list.append(transforms.RandAugment(num_ops=2, magnitude=randaugment_magnitude))
    transform_list.extend(
        [
            transforms.RandomAffine(degrees=12, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
        ]
    )
    return transforms.Compose(transform_list)


def build_eval_transform(image_size):
    resize_size = int(image_size * 256 / 224)
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_dataloader(dataset, batch_size, num_workers, shuffle, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=num_workers > 0,
    )


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def remap_checkpoint_keys(state_dict, model_name):
    if model_name.startswith("eca_"):
        remapped = {}
        for key, value in state_dict.items():
            if ".eca." in key:
                remapped[key.replace(".eca.", ".attention.")] = value
            else:
                remapped[key] = value
        return remapped
    return state_dict


def load_model_weights(model, checkpoint_path, device, model_name):
    state_dict = load_checkpoint(checkpoint_path, device)
    state_dict = remap_checkpoint_keys(state_dict, model_name)
    model.load_state_dict(state_dict)
    return state_dict


def build_eval_dataset(data_dir, image_size, split):
    eval_transform = build_eval_transform(image_size)
    if split == "val":
        return ImageFolder(data_dir / "val", transform=eval_transform)
    return TestDataset(root=data_dir / "test", transform=eval_transform)


def normalize_ensemble_members(model_names, model_paths):
    if not model_paths:
        return []
    if not model_names:
        raise ValueError("ensemble-models must be provided when ensemble-model-paths are specified.")

    normalized_model_names = list(model_names)
    normalized_model_paths = list(model_paths)
    if len(normalized_model_names) == 1 and len(normalized_model_paths) > 1:
        normalized_model_names *= len(normalized_model_paths)
    if len(normalized_model_names) != len(normalized_model_paths):
        raise ValueError("ensemble-models and ensemble-model-paths must have equal length.")
    return list(zip(normalized_model_names, normalized_model_paths))


def infer_model_name_from_path(checkpoint_path, default_model):
    stem = checkpoint_path.stem
    for model_name in sorted(MODEL_CHOICES, key=len, reverse=True):
        if stem == model_name or stem.startswith(f"{model_name}_"):
            return model_name
    return default_model


def infer_image_size_from_state_dict(state_dict, model_name, default_image_size):
    if not model_name.startswith("gmlp_"):
        return default_image_size

    proj_weight = state_dict.get("gmlp_blocks.0.sgu.proj.weight")
    if proj_weight is None or proj_weight.ndim != 2 or proj_weight.shape[0] != proj_weight.shape[1]:
        return default_image_size

    seq_len = proj_weight.shape[0]
    token_grid = int(round(math.sqrt(seq_len)))
    if token_grid * token_grid != seq_len:
        return default_image_size
    return token_grid * 32


def discover_ensemble_candidates(args):
    if args.ensemble_model_paths:
        return normalize_ensemble_members(args.ensemble_models, args.ensemble_model_paths)

    pattern = args.ensemble_candidate_glob
    candidate_paths = [Path(path).resolve() for path in glob.glob(pattern)]
    if not candidate_paths and not Path(pattern).is_absolute():
        candidate_paths = [path.resolve() for path in PROJECT_ROOT.glob(pattern)]
    candidate_paths = sorted(set(candidate_paths))
    if not candidate_paths:
        raise ValueError(f"No ensemble candidate checkpoints matched: {pattern}")

    if args.ensemble_models:
        model_names = list(args.ensemble_models)
        if len(model_names) == 1:
            model_names *= len(candidate_paths)
        elif len(model_names) != len(candidate_paths):
            raise ValueError("When using ensemble-candidate-glob, ensemble-models must be length 1 or match the number of discovered checkpoints.")
    else:
        model_names = [infer_model_name_from_path(path, args.model) for path in candidate_paths]
    return list(zip(model_names, candidate_paths))


def resolve_ensemble_member_specs(args):
    member_specs = []
    for model_name, model_path in discover_ensemble_candidates(args):
        resolved_path = model_path.resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"Ensemble checkpoint not found: {resolved_path}")
        state_dict = load_checkpoint(resolved_path, args.device)
        state_dict = remap_checkpoint_keys(state_dict, model_name)
        image_size = infer_image_size_from_state_dict(state_dict, model_name, args.image_size)
        member_specs.append(
            {
                "model_name": model_name,
                "model_path": resolved_path,
                "image_size": image_size,
            }
        )
    return member_specs


def select_top_ensemble_members(args, train_dataset):
    if args.ensemble_top_k is None:
        return resolve_ensemble_member_specs(args)
    if args.ensemble_top_k <= 0:
        raise ValueError("ensemble-top-k must be greater than 0.")

    candidate_members = resolve_ensemble_member_specs(args)
    val_dataloaders = {}

    scored_members = []
    for member in candidate_members:
        model_name = member["model_name"]
        resolved_path = member["model_path"]
        image_size = member["image_size"]
        if image_size not in val_dataloaders:
            val_dataset = build_eval_dataset(args.data_dir, image_size, split="val")
            val_dataloaders[image_size] = build_dataloader(
                val_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                device=args.device,
            )
        val_dataloader = val_dataloaders[image_size]
        model_args = argparse.Namespace(**vars(args))
        model_args.model = model_name
        model_args.image_size = image_size
        model = build_model(model_args, pretrained=False)
        load_model_weights(model, resolved_path, args.device, model_name)
        trainer = Trainer(
            model,
            None,
            None,
            test_dataloader=val_dataloader,
            device=args.device,
            class_names=train_dataset.classes,
            tta_horizontal_flip=args.tta_horizontal_flip,
        )
        logits, labels = trainer.predict_labeled_logits(val_dataloader)
        val_loss = torch.nn.functional.cross_entropy(logits, labels).item()
        val_acc = (logits.argmax(dim=1) == labels).float().mean().item()
        scored_members.append((val_acc, val_loss, model_name, resolved_path, image_size))
        print(
            f"Validated ensemble candidate: {model_name} from {resolved_path} "
            f"| image_size={image_size}, val_acc={val_acc:.4%}, val_loss={val_loss:.4f}"
        )

    scored_members.sort(key=lambda item: (-item[0], item[1], str(item[3])))
    top_k = min(args.ensemble_top_k, len(scored_members))
    selected_members = [
        {
            "model_name": model_name,
            "model_path": model_path,
            "image_size": image_size,
        }
        for _, _, model_name, model_path, image_size in scored_members[:top_k]
    ]
    print("Selected ensemble members:")
    for rank, (val_acc, val_loss, model_name, model_path, image_size) in enumerate(scored_members[:top_k], start=1):
        print(
            f"  Top {rank}: {model_name} from {model_path} "
            f"| image_size={image_size}, val_acc={val_acc:.4%}, val_loss={val_loss:.4f}"
        )
    return selected_members


def build_scheduler(optimizer, epochs, warmup_ratio, eta_min):
    warmup_epochs = min(epochs - 1, max(0, math.ceil(epochs * warmup_ratio)))
    if warmup_epochs == 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=eta_min)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=eta_min)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )


def init_wandb_run(args, run_type, model_path=None, extra_config=None):
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is not installed. Install it first to run this project.") from exc

    config = {
        "mode": args.mode,
        "run_type": run_type,
        "model": args.model,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "backbone_lr_scale": args.backbone_lr_scale,
        "optimizer": args.optimizer,
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "warmup_ratio": args.warmup_ratio,
        "eta_min": args.eta_min,
        "image_size": args.image_size,
        "randaugment_magnitude": args.randaugment_magnitude,
        "mixup_alpha": args.mixup_alpha,
        "cutmix_alpha": args.cutmix_alpha,
        "mix_prob": args.mix_prob,
        "seed": args.seed,
        "device": args.device,
        "data_dir": str(args.data_dir),
        "output_path": str(args.output_path),
        "freeze_backbone": args.freeze_backbone,
        "resume_train": args.resume_train,
        "tta_horizontal_flip": args.tta_horizontal_flip,
    }
    if model_path is not None:
        config["model_path"] = str(model_path)
    if extra_config is not None:
        config.update(extra_config)

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        tags=args.wandb_tags,
        config=config,
    )


def build_model(args, pretrained=True):
    return Resnet(
        num_classes=100,
        model_name=args.model,
        freeze=args.freeze_backbone,
        pretrained=pretrained,
        dropout=args.dropout,
        image_size=args.image_size,
    )


def build_optimizer(model, optimizer_name, lr, weight_decay, backbone_lr_scale):
    parameter_groups = model.parameter_groups(lr, backbone_lr_scale=backbone_lr_scale)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(parameter_groups, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            parameter_groups,
            lr=lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def train(args, model_path, wandb_run=None):
    model = build_model(args, pretrained=not args.resume_train)
    if args.resume_train:
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found for resume training: {model_path}")
        load_model_weights(model, model_path, args.device, args.model)
        print(f"Resumed training from {model_path}")

    optimizer = build_optimizer(
        model,
        args.optimizer,
        args.lr,
        args.weight_decay,
        args.backbone_lr_scale,
    )
    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_ratio, args.eta_min)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    train_dataset = ImageFolder(
        args.data_dir / "train",
        transform=build_train_transform_with_randaugment(args.image_size, args.randaugment_magnitude),
    )
    val_dataset = ImageFolder(args.data_dir / "val", transform=build_eval_transform(args.image_size))
    train_dataloader = build_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        device=args.device,
    )
    val_dataloader = build_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        device=args.device,
    )

    trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=args.device,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        mix_prob=args.mix_prob,
        wandb_run=wandb_run,
    )
    metrics = trainer.train(epochs=args.epochs, save_path=model_path)
    if wandb_run is not None:
        wandb_run.summary["saved_model_path"] = str(model_path)
    print(
        f"Best model saved at {model_path} "
        f"(epoch {metrics['best_epoch']}, val acc {metrics['best_val_acc']:.4%}, "
        f"val loss {metrics['best_val_loss']:.4f})"
    )


def test(args, model_path, wandb_run=None):
    model = build_model(args, pretrained=False)
    load_model_weights(model, model_path, args.device, args.model)
    train_dataset = ImageFolder(args.data_dir / "train")
    test_dataset = TestDataset(root=args.data_dir / "test", transform=build_eval_transform(args.image_size))
    test_dataloader = build_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        device=args.device,
    )
    trainer = Trainer(
        model,
        None,
        None,
        test_dataloader=test_dataloader,
        device=args.device,
        class_names=train_dataset.classes,
        tta_horizontal_flip=args.tta_horizontal_flip,
        wandb_run=wandb_run,
    )
    trainer.test(test_dataloader, output_path=args.output_path)


def ensemble_test(args, wandb_run=None):
    train_dataset = ImageFolder(args.data_dir / "train")
    selected_members = select_top_ensemble_members(args, train_dataset)
    eval_dataloaders = {}

    ensemble_logits = None
    ensemble_filenames = None
    ensemble_labels = None
    for member in selected_members:
        model_name = member["model_name"]
        resolved_path = member["model_path"].resolve()
        image_size = member["image_size"]
        if image_size not in eval_dataloaders:
            eval_dataset = build_eval_dataset(args.data_dir, image_size, split=args.ensemble_split)
            eval_dataloaders[image_size] = build_dataloader(
                eval_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                device=args.device,
            )
        eval_dataloader = eval_dataloaders[image_size]
        model_args = argparse.Namespace(**vars(args))
        model_args.model = model_name
        model_args.image_size = image_size
        model = build_model(model_args, pretrained=False)
        load_model_weights(model, resolved_path, args.device, model_name)
        trainer = Trainer(
            model,
            None,
            None,
            test_dataloader=eval_dataloader,
            device=args.device,
            class_names=train_dataset.classes,
            tta_horizontal_flip=args.tta_horizontal_flip,
        )
        if args.ensemble_split == "val":
            logits, labels = trainer.predict_labeled_logits(eval_dataloader)
            if ensemble_labels is None:
                ensemble_labels = labels
                ensemble_logits = logits
            else:
                if not torch.equal(labels, ensemble_labels):
                    raise ValueError("Ensemble label order mismatch across models.")
                ensemble_logits += logits
        else:
            filenames, logits = trainer.predict_logits(eval_dataloader)
            if ensemble_filenames is None:
                ensemble_filenames = filenames
                ensemble_logits = logits
            else:
                if filenames != ensemble_filenames:
                    raise ValueError("Ensemble dataloader order mismatch across models.")
                ensemble_logits += logits
        print(f"Loaded ensemble member: {model_name} from {resolved_path} | image_size={image_size}")

    ensemble_logits /= len(selected_members)
    if args.ensemble_split == "val":
        val_loss = torch.nn.functional.cross_entropy(ensemble_logits, ensemble_labels).item()
        val_acc = (ensemble_logits.argmax(dim=1) == ensemble_labels).float().mean().item()
        if wandb_run is not None:
            wandb_run.log({"ensemble/val_loss": val_loss, "ensemble/val_acc": val_acc})
            wandb_run.summary["ensemble_num_models"] = len(selected_members)
        print(f"Ensemble validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4%} " f"using {len(selected_members)} models.")
        return

    pred_indices = torch.argmax(ensemble_logits, dim=1).tolist()
    preds = [int(train_dataset.classes[idx]) for idx in pred_indices]
    output_rows = []
    for filename, pred in zip(ensemble_filenames, preds):
        image_name = Path(filename).stem
        output_rows.append((image_name, pred))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_name", "pred_label"])
        writer.writerows(output_rows)
    if wandb_run is not None:
        wandb_run.summary["prediction_output_path"] = str(args.output_path)
        wandb_run.summary["ensemble_num_models"] = len(selected_members)
    print(f"Saved ensemble predictions to {args.output_path} " f"using {len(selected_members)} models.")


def tune(args, model_path, wandb_run=None):
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Optuna is not installed. Install it first to use mode=tune.") from exc

    tune_dir = model_path.parent / "optuna_trials"
    tune_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        trial_seed = args.seed + trial.number
        set_seed(trial_seed)
        trial_args = argparse.Namespace(**vars(args))
        trial_args.dropout = trial.suggest_float("dropout", 0.1, 0.5)
        trial_args.optimizer = trial.suggest_categorical("optimizer", ["adamw", "sgd"])
        trial_args.lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        trial_args.batch_size = trial.suggest_categorical("batch_size", [256])
        trial_args.epochs = trial.suggest_int("epochs", 20, 30)
        trial_args.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        trial_args.backbone_lr_scale = trial.suggest_categorical("backbone_lr_scale", [0.01, 0.05, 0.1, 0.2])
        trial_args.warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
        trial_args.eta_min = trial.suggest_float("eta_min", 1e-7, 1e-5, log=True)
        trial_args.label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
        trial_args.randaugment_magnitude = trial.suggest_int("randaugment_magnitude", 5, 15)
        trial_args.image_size = trial.suggest_categorical("image_size", [224, 256, 288])

        train_dataset = ImageFolder(
            trial_args.data_dir / "train",
            transform=build_train_transform_with_randaugment(trial_args.image_size, trial_args.randaugment_magnitude),
        )
        val_dataset = ImageFolder(
            trial_args.data_dir / "val",
            transform=build_eval_transform(trial_args.image_size),
        )

        model = build_model(trial_args, pretrained=True)
        optimizer = build_optimizer(
            model,
            trial_args.optimizer,
            trial_args.lr,
            trial_args.weight_decay,
            trial_args.backbone_lr_scale,
        )
        scheduler = build_scheduler(optimizer, trial_args.epochs, trial_args.warmup_ratio, trial_args.eta_min)
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=trial_args.label_smoothing)

        train_dataloader = build_dataloader(
            train_dataset,
            batch_size=trial_args.batch_size,
            num_workers=trial_args.num_workers,
            shuffle=True,
            device=trial_args.device,
        )
        val_dataloader = build_dataloader(
            val_dataset,
            batch_size=trial_args.batch_size,
            num_workers=trial_args.num_workers,
            shuffle=False,
            device=trial_args.device,
        )

        trainer = Trainer(
            model,
            optimizer,
            loss_fn,
            scheduler=scheduler,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            device=trial_args.device,
            early_stopping_patience=trial_args.early_stopping_patience,
            early_stopping_min_delta=trial_args.early_stopping_min_delta,
            mixup_alpha=trial_args.mixup_alpha,
            cutmix_alpha=trial_args.cutmix_alpha,
            mix_prob=trial_args.mix_prob,
        )
        trial_model_path = tune_dir / f"{model_path.stem}_trial_{trial.number}.pth"
        metrics = trainer.train(epochs=trial_args.epochs, save_path=trial_model_path)
        trial.set_user_attr("model_path", str(trial_model_path))
        trial.set_user_attr("seed", trial_seed)
        trial.set_user_attr("best_epoch", metrics["best_epoch"])
        if wandb_run is not None:
            wandb_run.log(
                {
                    "optuna/trial": trial.number,
                    "optuna/best_val_acc": metrics["best_val_acc"],
                    "optuna/best_val_loss": metrics["best_val_loss"],
                    "optuna/best_epoch": metrics["best_epoch"],
                }
            )
        return metrics["best_val_acc"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.optuna_trials, timeout=args.optuna_timeout)

    best_model_path = Path(study.best_trial.user_attrs["model_path"])
    shutil.copy2(best_model_path, model_path)
    if wandb_run is not None:
        wandb_run.summary["optuna_best_val_acc"] = study.best_value
        wandb_run.summary["optuna_best_epoch"] = study.best_trial.user_attrs["best_epoch"]
        wandb_run.summary["optuna_best_seed"] = study.best_trial.user_attrs["seed"]
        wandb_run.summary["saved_model_path"] = str(model_path)
        for key, value in study.best_params.items():
            wandb_run.summary[f"best_param/{key}"] = value

    print(f"Best trial val acc: {study.best_value:.4%}")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    print(f"best_epoch: {study.best_trial.user_attrs['best_epoch']}")
    print(f"seed: {study.best_trial.user_attrs['seed']}")
    print(f"Best trial checkpoint: {best_model_path}")
    print(f"Copied best checkpoint to: {model_path}")


def main():
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.output_path = args.output_path.resolve()
    args.wandb_tags = parse_wandb_tags(args.wandb_tags)
    if args.ensemble_model_paths is not None:
        args.ensemble_model_paths = [path.resolve() for path in args.ensemble_model_paths]
    model_path = args.model_path.resolve() if args.model_path is not None else (PROJECT_ROOT / "models" / f"{args.model}.pth").resolve()

    set_seed(args.seed)

    if args.mode in {"train", "all"}:
        train_run = init_wandb_run(args, "train", model_path=model_path)
        try:
            train(args, model_path, wandb_run=train_run)
        finally:
            if train_run is not None:
                train_run.finish()
    if args.mode in {"test", "all"}:
        extra_config = None
        if args.ensemble_models or args.ensemble_model_paths or args.ensemble_top_k is not None:
            extra_config = {
                "ensemble_models": args.ensemble_models,
                "ensemble_model_paths": [str(path) for path in args.ensemble_model_paths] if args.ensemble_model_paths else None,
                "ensemble_top_k": args.ensemble_top_k,
                "ensemble_candidate_glob": args.ensemble_candidate_glob,
                "ensemble_split": args.ensemble_split,
            }
        test_run = init_wandb_run(args, "test", model_path=model_path, extra_config=extra_config)
        if args.ensemble_models or args.ensemble_model_paths or args.ensemble_top_k is not None:
            try:
                ensemble_test(args, wandb_run=test_run)
            finally:
                if test_run is not None:
                    test_run.finish()
        else:
            try:
                test(args, model_path, wandb_run=test_run)
            finally:
                if test_run is not None:
                    test_run.finish()
    if args.mode == "tune":
        tune_run = init_wandb_run(
            args,
            "tune",
            model_path=model_path,
            extra_config={
                "optuna_trials": args.optuna_trials,
                "optuna_timeout": args.optuna_timeout,
            },
        )
        try:
            tune(args, model_path, wandb_run=tune_run)
        finally:
            if tune_run is not None:
                tune_run.finish()


if __name__ == "__main__":
    main()
