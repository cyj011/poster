"""Train POSTER on FER2013 arranged as train/val/test class folders.

Expected layout:

    /mnt/data/yanyi2025/cyj/fer2013_img/
        train/<class_name>/*.jpg
        val/<class_name>/*.jpg
        test/<class_name>/*.jpg

The best checkpoint is selected on the validation split and evaluated once on
the test split at the end of training.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/data/yanyi2025/cyj/fer2013_img"),
        help="FER2013 root containing train, val and test directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("result/fer2013"),
        help="Directory for checkpoints, metrics and logs.",
    )
    parser.add_argument(
        "--gpu",
        default="0",
        help="Physical GPU id to use. Use 'cpu' to disable CUDA.",
    )
    parser.add_argument("--modeltype", choices=["small", "base", "large"], default="large")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--sam-rho", type=float, default=0.05)
    parser.add_argument("--scheduler-gamma", type=float, default=0.98)
    parser.add_argument(
        "--balanced-sampler",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use inverse-frequency sampling for the imbalanced FER2013 train split.",
    )
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


args = parse_args()

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if args.gpu.lower() == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from data_preprocessing.sam import SAM
from models.emotion_hyp import pyramid_trans_expr
from torchsampler import ImbalancedDatasetSampler
from utils import LabelSmoothingCrossEntropy


IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_transforms():
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(scale=(0.02, 0.1)),
    ])
    eval_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, eval_transform


def load_bgr_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Unable to read image: {path}")
    return image


def make_loader(dataset, batch_size, shuffle, workers, pin_memory, sampler=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        drop_last=False,
    )


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    predictions = []
    targets_all = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs, _ = model(images)
            loss = criterion(outputs, targets)
            predicts = outputs.argmax(dim=1)

            total_loss += loss.item() * targets.size(0)
            correct += (predicts == targets).sum().item()
            total += targets.size(0)
            predictions.extend(predicts.cpu().tolist())
            targets_all.extend(targets.cpu().tolist())

    return (
        total_loss / total,
        correct / total,
        f1_score(targets_all, predictions, average="macro"),
    )


def save_checkpoint(path, model, optimizer, scheduler, epoch, val_acc, class_names):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": val_acc,
            "class_names": class_names,
            "modeltype": args.modeltype,
        },
        path,
    )


def main():
    set_seed(args.seed)
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    missing_splits = [
        split for split in ("train", "val", "test")
        if not (data_root / split).is_dir()
    ]
    if missing_splits:
        raise FileNotFoundError(
            f"Missing FER2013 directories under {data_root}: "
            f"{', '.join(missing_splits)}"
        )

    required_weights = [
        REPO_ROOT / "models" / "pretrain" / "ir50.pth",
        REPO_ROOT / "models" / "pretrain" / "mobilefacenet_model_best.pth.tar",
    ]
    missing_weights = [str(path) for path in required_weights if not path.is_file()]
    if missing_weights:
        raise FileNotFoundError(
            "Missing POSTER backbone weights. Place these files here:\n  "
            + "\n  ".join(missing_weights)
        )

    if args.gpu.lower() == "cpu":
        device = torch.device("cpu")
    elif not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Check the NVIDIA driver and PyTorch installation."
        )
    else:
        device = torch.device("cuda:0")

    train_transform, eval_transform = make_transforms()
    train_dataset = datasets.ImageFolder(
        data_root / "train",
        transform=train_transform,
        loader=load_bgr_image,
    )
    val_dataset = datasets.ImageFolder(
        data_root / "val",
        transform=eval_transform,
        loader=load_bgr_image,
    )
    test_dataset = datasets.ImageFolder(
        data_root / "test",
        transform=eval_transform,
        loader=load_bgr_image,
    )

    if not (
        train_dataset.class_to_idx
        == val_dataset.class_to_idx
        == test_dataset.class_to_idx
    ):
        raise ValueError(
            "Class folders differ between splits: "
            f"train={train_dataset.class_to_idx}, "
            f"val={val_dataset.class_to_idx}, "
            f"test={test_dataset.class_to_idx}"
        )

    class_names = train_dataset.classes
    if len(class_names) != 7:
        raise ValueError(
            f"Expected 7 FER2013 emotion classes, found {len(class_names)}: "
            f"{class_names}"
        )

    pin_memory = device.type == "cuda"
    sampler = (
        ImbalancedDatasetSampler(train_dataset)
        if args.balanced_sampler
        else None
    )
    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        shuffle=sampler is None,
        workers=args.workers,
        pin_memory=pin_memory,
        sampler=sampler,
    )
    val_loader = make_loader(
        val_dataset,
        args.val_batch_size,
        shuffle=False,
        workers=args.workers,
        pin_memory=pin_memory,
    )
    test_loader = make_loader(
        test_dataset,
        args.val_batch_size,
        shuffle=False,
        workers=args.workers,
        pin_memory=pin_memory,
    )

    model = pyramid_trans_expr(
        img_size=IMAGE_SIZE,
        num_classes=len(class_names),
        type=args.modeltype,
    ).to(device)
    optimizer = SAM(
        model.parameters(),
        torch.optim.Adam,
        lr=args.lr,
        rho=args.sam_rho,
        adaptive=False,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=args.scheduler_gamma
    )
    ce_criterion = torch.nn.CrossEntropyLoss()
    lsce_criterion = LabelSmoothingCrossEntropy(smoothing=0.2)

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(
            f"CUDA device: {torch.cuda.get_device_name(0)} "
            f"(physical GPU {args.gpu})"
        )
    print(f"Dataset: {data_root}")
    print(f"Classes: {json.dumps(class_names, ensure_ascii=True)}")
    print(
        f"Samples: train={len(train_dataset)}, "
        f"val={len(val_dataset)}, test={len(test_dataset)}"
    )
    print(f"Model: POSTER-{args.modeltype}")
    print(f"Batch size: {args.batch_size}, balanced sampler: {args.balanced_sampler}")

    history = []
    best_val_acc = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = (
                ce_criterion(outputs, targets)
                + 2.0 * lsce_criterion(outputs, targets)
            )
            loss.backward()
            optimizer.first_step(zero_grad=True)

            outputs, _ = model(images)
            loss = (
                ce_criterion(outputs, targets)
                + 2.0 * lsce_criterion(outputs, targets)
            )
            loss.backward()
            optimizer.second_step(zero_grad=True)

            train_loss_sum += loss.item() * targets.size(0)
            train_correct += (outputs.argmax(dim=1) == targets).sum().item()
            train_total += targets.size(0)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total
        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, ce_criterion, device
        )
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
                "learning_rate": current_lr,
            }
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_loss:.4f}, acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f}, acc {val_acc:.4f}, "
            f"f1 {val_f1:.4f} | lr {current_lr:.2e}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_checkpoint(
                output_dir / "best_model.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                val_acc,
                class_names,
            )
            print(f"  Saved best checkpoint (val acc={val_acc:.4f})")

    best_checkpoint = torch.load(
        output_dir / "best_model.pth",
        map_location=device,
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_loss, test_acc, test_f1 = evaluate(
        model, test_loader, ce_criterion, device
    )

    metrics = {
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "class_names": class_names,
        "modeltype": args.modeltype,
        "data_root": str(data_root),
        "balanced_sampler": args.balanced_sampler,
    }
    pd.DataFrame(history).to_csv(output_dir / "metrics.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "class_names.json").write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Best epoch {best_epoch} | val acc {best_val_acc:.4f} | "
        f"test acc {test_acc:.4f} | test macro-F1 {test_f1:.4f}"
    )
    print(f"Saved checkpoint to {output_dir / 'best_model.pth'}")
    print(f"Saved metrics to {output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
