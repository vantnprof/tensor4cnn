import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    


class MultilinearTransformationLayer(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(MultilinearTransformationLayer, self).__init__()
        D1, D2, D3 = input_shape
        d1, d2, d3 = output_shape
        self.U1 = nn.Parameter(torch.randn(d1, D1))
        self.U2 = nn.Parameter(torch.randn(d2, D2))
        self.U3 = nn.Parameter(torch.randn(d3, D3))

    def forward(self, X):
        X = torch.tensordot(self.U1, X, dims=([1],[1])).permute(1, 0, 2, 3)
        X = torch.tensordot(self.U2, X, dims=([1],[2])).permute(1, 2, 0, 3)
        X = torch.tensordot(self.U3, X, dims=([1],[3])).permute(1, 2, 3, 0)
        return X
    

class SimpleCNNWithMTL(nn.Module):
    def __init__(self):
        super(SimpleCNNWithMTL, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.mtl = MultilinearTransformationLayer((32, 8, 8), (64, 8, 8))
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.mtl(x)
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


train_dataset = torchvision.datasets.CIFAR10(root='./data', 
                                             train=True, download=True, 
                                             transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', 
                                            train=False, download=True, 
                                            transform=transform)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, 
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, 
                                          shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Training on: {device}')


def train(model, criterion, optimizer, loader, epochs=10):
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss / len(loader))
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(loader):.4f}')
    return losses


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


model_mtl = SimpleCNNWithMTL().to(device)
model_pure = SimpleCNN().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_mtl.parameters(), lr=0.001)
optimizer_pure = torch.optim.Adam(model_pure.parameters(), lr=0.001)


losses_pure = train(model_pure, criterion, optimizer_pure, train_loader, epochs=100)
losses_mtl = train(model_mtl, criterion, optimizer, train_loader, epochs=100)


plt.plot(losses_mtl, label='CNN with MTL')
plt.plot(losses_pure, label='Pure CNN')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss on CIFAR10')
plt.legend()
plt.grid()
plt.savefig('losses.png')


train_acc_mtl = evaluate(model_mtl, train_loader)
test_acc_mtl = evaluate(model_mtl, test_loader)
train_acc_pure = evaluate(model_pure, train_loader)
test_acc_pure = evaluate(model_pure, test_loader)

print(f"CNN with MTL - Train Accuracy: {train_acc_mtl:.2f}%, Test Accuracy: {test_acc_mtl:.2f}%")
print(f"Pure CNN     - Train Accuracy: {train_acc_pure:.2f}%, Test Accuracy: {test_acc_pure:.2f}%")