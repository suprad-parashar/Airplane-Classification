import torch.nn as nn
import torch.optim as optim
import torch
from data import AircraftDataset
from tqdm import tqdm
import torchvision.models as models
    
class PreTrainedResnetCNN(nn.Module):
    def __init__(self, num_classes):
        super(PreTrainedResnetCNN, self).__init__()
        self.model = models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        features = self.model(x)
        return features

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, num_classes):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.channels = [3, 4, 16, 32, 64, 128]
        self.cnns = nn.ModuleList([self.get_cnn_layer(self.channels[i], self.channels[i+1], 3, 1, 0) for i in range(len(self.channels)-1)])
        self.fc = nn.Sequential(
            nn.Linear(3 * 3 * 128, 256),
            nn.Linear(256, num_classes)
        )
    def get_cnn_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def forward(self, x):
        for cnn in self.cnns:
            x = cnn(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    
    dataset = AircraftDataset()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = PreTrainedResnetCNN(dataset.NUM_CLASSES)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.005)

    for param in model.model.parameters():
        param.requires_grad = False
    for param in model.model.fc.parameters():
        param.requires_grad = True

    print("Training the model...")
    
    train_dataloader = dataset.get_dataloader("train")
    val_dataloader = dataset.get_dataloader("val")
    epochs = 50
    for epoch in range(epochs):
        model.train()
        for (images, labels) in tqdm(train_dataloader):
            outputs = model(images.to(device))
            loss = criterion(outputs, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for (images, labels) in tqdm(val_dataloader):
                outputs = model(images.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()
            print('Accuracy: {} %'.format(100 * correct / total))

    print("Testing the model...")
    test_dataloader = dataset.get_dataloader("test")
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for (images, labels) in tqdm(test_dataloader):
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
        print('Accuracy: {} %'.format(100 * correct / total))

    torch.save(model, "Models/cnn_resnet.pt")