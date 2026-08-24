import argparse
import gzip
import struct
import time
import urllib.request
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download_mnist(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in FILES.values():
        path = data_dir / filename
        if path.exists():
            continue
        url = MNIST_URL + filename
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)


def read_images(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image file magic number {magic} in {path}")
        data = torch.frombuffer(bytearray(f.read()), dtype=torch.uint8).clone()
    return data.view(count, 1, rows, cols).float().div_(255.0)


def read_labels(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label file magic number {magic} in {path}")
        data = torch.frombuffer(bytearray(f.read()), dtype=torch.uint8).clone()
    return data.long().view(count)


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.numel()
    return total_loss / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small PyTorch CNN on MNIST.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/mnist"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mnist"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.threads < 1:
        parser.error("--threads must be at least 1")

    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    download_mnist(args.data_dir)
    train_images = read_images(args.data_dir / FILES["train_images"])
    train_labels = read_labels(args.data_dir / FILES["train_labels"])
    test_images = read_images(args.data_dir / FILES["test_images"])
    test_labels = read_labels(args.data_dir / FILES["test_labels"])

    train_loader = DataLoader(
        TensorDataset(train_images, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(test_images, test_labels),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Training on {device} for {args.epochs} epoch(s)")
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.numel()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.numel()

        test_loss, test_acc = evaluate(model, test_loader, device)
        print(
            f"epoch {epoch:02d} "
            f"train_loss={total_loss / total:.4f} "
            f"train_acc={correct / total:.4f} "
            f"test_loss={test_loss:.4f} "
            f"test_acc={test_acc:.4f}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "mnist_cnn.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "device": str(device),
            "test_loss": test_loss,
            "test_accuracy": test_acc,
        },
        model_path,
    )
    print(f"Saved model to {model_path}")
    print(f"Elapsed seconds: {time.time() - start:.1f}")


if __name__ == "__main__":
    main()
