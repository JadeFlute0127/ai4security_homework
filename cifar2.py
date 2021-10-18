import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

batch_size = 64

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4750, 0.4750, 0.4750], std=[0.2008, 0.2008, 0.2008])
])

train_data = datasets.CIFAR10(
    root='cifar10',
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.CIFAR10(
    root='cifar10',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.extra(x) + out
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(64)
        )

        self.block1 = ResBlock(64, 128, stride=2)
        self.block2 = ResBlock(128, 256, stride=2)
        self.block3 = ResBlock(256, 512, stride=2)
        self.block4 = ResBlock(512, 512, stride=2)
        self.fc = nn.Linear(512 * 1 * 1, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = torch.flatten(x, 1)
        out = self.fc(x)
        return out
model = ResNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
path = 'cifar10_state.pth'
if os.path.exists(path):
    model.load_state_dict(torch.load(path))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)


def train(model, data_loader, optimizer, epoch):
    model.train()
    for batch_idx, (x, label) in enumerate(data_loader):
        optimizer.zero_grad()
        x = x.cuda() #cuda
        output = model(x)
        label = label.cuda()#cuda
        loss = criterion(output, label).cuda()
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            print('epoch', epoch)
            print(batch_idx, 'loss:', loss.item())


def test(model, test_loader):
    Losses = []
    Acces = []

    model.eval()
    total_correct = 0.
    test_loss = 0.
    with torch.no_grad():
        for x, label in test_loader:
            x = x.cuda() #cuda
            output = model(x)
            label = label.cuda() #cuda
            test_loss += criterion(output, label)
            pred = output.argmax(dim=1)
            total_correct += pred.eq(label).sum().item()

        total_size = len(test_loader.dataset)
        test_loss /= total_size
        total_correct /= total_size
        Losses.append(test_loss)
        Acces.append(total_correct)
        print('test loss: ', test_loss)
        print('acc: ', total_correct * 100)
        return Losses,Acces

epochs = 5
iterations = []
for epoch in range(epochs):
    train(model, train_loader, optimizer, epoch)
    Losses ,Acces = test(model, test_loader)
    iterations.append(epoch)


torch.save(model.state_dict(), path)

###plt
plt.title('Loss And Acc')
plt.plot(iterations, Losses, color='darkblue',label='loss')
plt.plot(iterations, Acces, 'b' , label='accuracy')
plt.legend()
plt.xlabel('Iteration')
