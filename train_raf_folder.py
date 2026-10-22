"""Train POSTER on RAF-DB arranged as train/val class folders.

Expected layout:

    /mnt/data/yanyi2025/cyj/raf-db/
        train/<class_name>/*.jpg
        val/<class_name>/*.jpg

Required POSTER backbone weights:

    models/pretrain/ir50.pth
    models/pretrain/mobilefacenet_model_best.pth.tar

The script uses the validation split for model selection because this dataset
layout has no separate test directory.
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
        default=Path("/mnt/data/yanyi2025/cyj/raf-db"),
        help="RAF-DB root containing train and val directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("result/RAF-DB"),
        help="Directory for checkpoints, metrics and logs. Relative paths are resolved from the POSTER repository root.",
    )
    parser.add_argument(
        "--gpu",
        default="0",
        help="Physical GPU id to use. Use 'cpu' to disable CUDA.",
    )
    parser.add_argument("--modeltype", choices=["small", "base", "large"], default="large")
    parser.add_argument(
        "--head-type",
        choices=["linear", "dual-gate"],
        default="linear",
        help="Classifier head; dual-gate combines image and landmark CLS features.",
    )
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
        help="Use inverse-frequency sampling for RAF-DB's imbalanced classes.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint path to resume from, for example result/RAF-DB/latest_model.pth.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use CUDA automatic mixed precision to reduce memory use and speed up training.",
    )
    parser.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Average original and horizontally flipped images during validation.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Use 0 to train all epochs; otherwise stop after this many non-improving epochs.",
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

import numpy as np
import pandas as pd
import torch
import cv2
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
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, val_transform


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


def evaluate(model, loader, criterion, device, tta=False):
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
            if tta:
                flipped_outputs, _ = model(torch.flip(images, dims=(3,)))
                outputs = (outputs + flipped_outputs) * 0.5
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


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    val_acc,
    best_val_acc,
    best_epoch,
    class_names,
    history,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_acc": val_acc,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "class_names": class_names,
        "history": history,
        "modeltype": args.modeltype,
        "head_type": args.head_type,
        "amp": args.amp,
        "tta": args.tta,
    }
    path = Path(path)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, path)


def main():
    set_seed(args.seed)
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    missing_splits = [
        split for split in ("train", "val")
        if not (data_root / split).is_dir()
    ]
    if missing_splits:
        raise FileNotFoundError(
            f"Missing RAF-DB directories under {data_root}: "
            f"{', '.join(missing_splits)}"
        )

    required_weights = [
        REPO_ROOT / "models" / "pretrain" / "ir50.pth",
        REPO_ROOT / "models" / "pretrain" / "mobilefacenet_model_best.pth.tar",
    ]
    missing_weights = [str(path) for path in required_weights if not path.is_file()]
    if missing_weights:
        raise FileNotFoundError(
            "Missing POSTER backbone weights. Download the official pretrain "
            "files and place them here:\n  " + "\n  ".join(missing_weights)
        )

    if args.gpu.lower() == "cpu":
        device = torch.device("cpu")
    elif not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Check the NVIDIA driver and PyTorch installation."
        )
    else:
        device = torch.device("cuda:0")

    train_transform, val_transform = make_transforms()
    train_dataset = datasets.ImageFolder(
        data_root / "train",
        transform=train_transform,
        loader=load_bgr_image,
    )
    val_dataset = datasets.ImageFolder(
        data_root / "val",
        transform=val_transform,
        loader=load_bgr_image,
    )

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError(
            "Class folders differ between train and val: "
            f"train={train_dataset.class_to_idx}, val={val_dataset.class_to_idx}"
        )
    class_names = train_dataset.classes
    if len(class_names) != 7:
        raise ValueError(
            f"Expected 7 emotion classes, found {len(class_names)}: {class_names}"
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

    model = pyramid_trans_expr(
        img_size=IMAGE_SIZE,
        num_classes=len(class_names),
        type=args.modeltype,
        head_type=args.head_type,
    ).to(device)
    base_optimizer = torch.optim.Adam
    optimizer = SAM(
        model.parameters(),
        base_optimizer,
        lr=args.lr,
        rho=args.sam_rho,
        adaptive=False,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=args.scheduler_gamma
    )
    amp_enabled = args.amp and device.type == "cuda"
    if amp_enabled and (
        not hasattr(torch.cuda, "is_bf16_supported")
        or not torch.cuda.is_bf16_supported()
    ):
        print("BF16 is unsupported on this GPU; continuing with FP32.")
        amp_enabled = False
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
    print(f"Samples: train={len(train_dataset)}, val={len(val_dataset)}")
    print(f"Model: POSTER-{args.modeltype}")
    print(f"Classifier head: {args.head_type}")
    print(f"Batch size: {args.batch_size}, balanced sampler: {args.balanced_sampler}")
    print(f"AMP: {amp_enabled}, validation TTA: {args.tta}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params / 1_000_000:.3f}M")

    history = []
    best_val_acc = -1.0
    best_epoch = 0
    stale_epochs = 0
    start_epoch = 1

    if args.resume is not None:
        resume_path = args.resume.expanduser()
        if not resume_path.is_absolute():
            resume_path = REPO_ROOT / resume_path
        resume_path = resume_path.resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        checkpoint = torch.load(resume_path, map_location=device)
        if checkpoint.get("head_type", "linear") != args.head_type:
            raise ValueError(
                "Checkpoint head type differs from the requested model: "
                f"checkpoint={checkpoint.get('head_type', 'linear')}, "
                f"requested={args.head_type}"
            )
        if checkpoint.get("class_names", class_names) != class_names:
            raise ValueError(
                "Checkpoint class order differs from the current dataset: "
                f"checkpoint={checkpoint['class_names']}, current={class_names}"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_acc = float(checkpoint.get("best_val_acc", -1.0))
        best_epoch = int(checkpoint.get("best_epoch", checkpoint.get("epoch", 0)))
        history = checkpoint.get("history", [])
        print(
            f"Resumed from {resume_path} at epoch {start_epoch}. "
            f"Best val acc so far: {best_val_acc:.4f} at epoch {best_epoch}."
        )

    if start_epoch > args.epochs:
        raise ValueError(
            f"Resume checkpoint is already at epoch {start_epoch - 1}, "
            f"but --epochs is {args.epochs}."
        )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        model.face_landback.eval()

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                outputs, _ = model(images)
                loss = (
                    ce_criterion(outputs, targets)
                    + 2.0 * lsce_criterion(outputs, targets)
                )
            loss.backward()
            optimizer.first_step(zero_grad=True)

            with torch.cuda.amp.autocast(
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
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
            model, val_loader, ce_criterion, device, tta=args.tta
        )
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        score = 0.67 * val_f1 + 0.33 * val_acc

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
                "val_score": score,
                "learning_rate": current_lr,
            }
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_loss:.4f}, acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f}, acc {val_acc:.4f}, "
            f"f1 {val_f1:.4f}, score {score:.4f} | lr {current_lr:.2e}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                output_dir / "best_model.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                val_acc,
                best_val_acc,
                best_epoch,
                class_names,
                history,
            )
            print(f"  Saved best checkpoint (val acc={val_acc:.4f})")
        else:
            stale_epochs += 1

        save_checkpoint(
            output_dir / "latest_model.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            val_acc,
            best_val_acc,
            best_epoch,
            class_names,
            history,
        )

        if (
            args.early_stopping_patience > 0
            and stale_epochs >= args.early_stopping_patience
        ):
            print("Early stopping: validation accuracy did not improve.")
            break

    metrics = {
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "class_names": class_names,
        "modeltype": args.modeltype,
        "head_type": args.head_type,
        "data_root": str(data_root),
        "balanced_sampler": args.balanced_sampler,
        "amp": amp_enabled,
        "tta": args.tta,
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
    print(f"Best checkpoint epoch {best_epoch} | val acc {best_val_acc:.4f}")
    print(f"Saved checkpoint to {output_dir / 'best_model.pth'}")
    print(f"Saved metrics to {output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
