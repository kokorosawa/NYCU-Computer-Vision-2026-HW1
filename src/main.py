import argparse
import random
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
DEFAULT_MODEL = "resnet50"
DEFAULT_NUM_WORKERS = 8
DEFAULT_SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Train and test image classifier")
    parser.add_argument(
        "--mode",
        choices=("train", "test", "all"),
        default="all",
        help="Run training, testing, or both.",
    )
    parser.add_argument(
        "--model",
        choices=("resnet18", "resnet34", "resnet50"),
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
        "--image-size",
        type=int,
        default=224,
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
        "--output-path",
        type=Path,
        default=PROJECT_ROOT / "output" / "prediction.csv",
        help="Prediction csv output path.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use, e.g. cuda or cpu.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_train_transform(image_size):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
        ]
    )


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


def train(args, model_path):
    model = Resnet(num_classes=100, model_name=args.model, freeze=args.freeze_backbone)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    train_dataset = ImageFolder(
        args.data_dir / "train", transform=build_train_transform(args.image_size)
    )
    val_dataset = ImageFolder(
        args.data_dir / "val", transform=build_eval_transform(args.image_size)
    )
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
    )
    metrics = trainer.train(epochs=args.epochs, save_path=model_path)
    print(
        f"Best model saved at {model_path} "
        f"(epoch {metrics['best_epoch']}, val acc {metrics['best_val_acc']:.4%})"
    )


def test(args, model_path):
    model = Resnet(num_classes=100, model_name=args.model)
    model.load_state_dict(load_checkpoint(model_path, args.device))
    test_dataset = TestDataset(
        root=args.data_dir / "test", transform=build_eval_transform(args.image_size)
    )
    test_dataloader = build_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        device=args.device,
    )
    trainer = Trainer(model, None, None, test_dataloader=test_dataloader, device=args.device)
    trainer.test(test_dataloader, output_path=args.output_path)


def main():
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.output_path = args.output_path.resolve()
    model_path = (
        args.model_path.resolve()
        if args.model_path is not None
        else (PROJECT_ROOT / "models" / f"{args.model}.pth").resolve()
    )

    set_seed(args.seed)

    if args.mode in {"train", "all"}:
        train(args, model_path)
    if args.mode in {"test", "all"}:
        test(args, model_path)


if __name__ == "__main__":
    main()
