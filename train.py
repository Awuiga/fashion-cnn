import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

transform = transforms.ToTensor()
dataset = datasets.FashionMNIST(
    root = 'data',
    train = True,
    download = True,
    transform = transform
)

image, target = dataset[0]

generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(
    dataset,
    [0.8, 0.2],
    generator = generator
)

train_dataloader = DataLoader(train_dataset,
                              batch_size = 64,
                              shuffle = True)

val_dataloader = DataLoader(val_dataset,
                            batch_size = 64,
                            shuffle = False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Fashion(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
           nn.Conv2d(1, 16, kernel_size=3, padding=1),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size = 2, stride = 2),
           nn.Conv2d(16, 32, kernel_size = 3, padding = 1),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size = 2 , stride = 2),
           nn.Flatten(),
           nn.Linear(1568, 128),
           nn.ReLU(),
           nn.Linear(128, 10)
        )

    def forward(self, images):
        logits = self.layers(images)
        return(logits)

model = Fashion().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001)


model.train()
epochs = 5
total_loss = 0.0

for epoch in range(epochs):
    total_loss = 0.0
    for images, targets in train_dataloader:
        optimizer.zero_grad()
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)
    print('Epoch', epoch + 1, 'Avg_Loss', avg_loss)

model.eval()
val_total_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for images, targets in val_dataloader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        predictions = logits.argmax(dim = 1)
        loss = criterion(logits, targets)
        correct += (predictions == targets).sum().item()
        total += len(targets)
        val_total_loss += loss.item()

    avg_val_loss = val_total_loss / len(val_dataloader)
    accuracy = (correct / total) * 100

check_point = {
    'model_state_dict': model.state_dict(),
    'classes': dataset.classes
}

torch.save(check_point, 'fashion_cnn.pt')
