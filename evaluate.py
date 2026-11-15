"""Оценить доверенный checkpoint Fashion CNN без повторного обучения."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
from datetime import datetime, timezone, timedelta

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


# Та же архитектура, что в train.py и predict.py. Их импорт запускал бы
# обучение или загрузку данных, поэтому оценка использует отдельный класс.
class Fashion(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2, 2), nn.Flatten(),
            nn.Linear(1568, 128), nn.ReLU(), nn.Linear(128, 10)
        )

    def forward(self, images):
        return self.layers(images)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='fashion_cnn.pt')
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--output', default='reports/metrics.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = Fashion().to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    train_set = datasets.FashionMNIST(args.data_root, train=True, download=True,
                                      transform=transforms.ToTensor())
    test_set = datasets.FashionMNIST(args.data_root, train=False, download=True,
                                     transform=transforms.ToTensor())
    if checkpoint['classes'] != train_set.classes:
        raise ValueError('Классы checkpoint не совпали с Fashion-MNIST.')
    fit, val = random_split(train_set, [.8, .2], generator=torch.Generator().manual_seed(42))
    def measure(dataset):
        matrix = torch.zeros(10, 10, dtype=torch.long)
        loss_sum = 0.
        with torch.no_grad():
            for images, targets in DataLoader(dataset, batch_size=128, shuffle=False):
                images, targets = images.to(device), targets.to(device)
                logits = model(images)
                loss_sum += nn.functional.cross_entropy(logits, targets, reduction='sum').item()
                ids = targets * 10 + logits.argmax(dim=1)
                matrix += torch.bincount(ids.cpu(), minlength=100).reshape(10, 10)
        accuracy = matrix.diag().sum().item() / len(dataset)
        return {'samples': len(dataset), 'accuracy': accuracy,
                'cross_entropy_per_image': loss_sum / len(dataset),
                'confusion_matrix': matrix.tolist()}
    with Path(args.checkpoint).open('rb') as file:
        checkpoint_hash = hashlib.sha256(file.read()).hexdigest()
    report = {
        'evaluation_date': datetime.now(timezone(timedelta(hours=3))).date().isoformat(),
        'checkpoint_sha256': checkpoint_hash,
        'parameters': sum(p.numel() for p in model.parameters()),
        'classes': checkpoint['classes'], 'train_samples': len(fit),
        'validation': measure(val), 'test': measure(test_set),
        'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                        'torchvision': torchvision.__version__, 'device': str(device)},
        'notes': ['Validation split reconstructed with seed 42 from supplied train.py.',
                  'Checkpoint does not record training indices, epoch or dataset hashes.',
                  'Metrics evaluate existing weights; no training or tuning is performed.',
                  'Loss is weighted per image, unlike the mean of batch means in train.py.']
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    for split in ['validation', 'test']:
        print(split, report[split]['samples'], 'accuracy', report[split]['accuracy'],
              'loss', report[split]['cross_entropy_per_image'], flush=True)
    print('parameters', report['parameters'])


if __name__ == '__main__':
    main()
