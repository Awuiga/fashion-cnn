import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

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

check_point = torch.load('fashion_cnn.pt',
                         map_location = 'cpu',
                         weights_only = True)
model = Fashion()
model.load_state_dict(check_point['model_state_dict'])
classes = check_point['classes']

model.eval()
transform = transforms.ToTensor()
test_dataset = datasets.FashionMNIST(root = 'data',
                                     train = False,
                                     download = True,
                                     transform = transform)
image, target = test_dataset[0]

with torch.no_grad():
    image = image.unsqueeze(0)
    logits = model(image)
    predictions = logits.argmax(dim = 1)

prediction_id = predictions.item()
predicted_class = classes[prediction_id]
correct_answer = classes[target]

print("Prediction:", predicted_class)
print("Target:", correct_answer)
