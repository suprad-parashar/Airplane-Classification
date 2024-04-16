from data_loader import get_test_dataloader, get_training_dataloader, get_validation_dataloader, classes
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, num_classes):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3), # 224x224x3 -> 222x222x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 222x222x32 -> 111x111x32
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3), # 111x111x32 -> 109x109x64
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2) # 109x109x64 -> 54x54x64
        )
        self.drop_out = nn.Dropout() 
        self.fc1 = nn.Linear(54 * 54 * 64, 1000)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out
    
if __name__ == "__main__":

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Create a DataLoader - Train, Test, Validation
    train_dataloader = get_training_dataloader()
    test_dataloader = get_test_dataloader()
    val_dataloader = get_validation_dataloader()
    
    # Create a Convolutional Neural Network Model
    model = ConvolutionalNeuralNetwork(len(classes)).to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-6)

    # Train the Model
    for epoch in range(10):
        for images, labels in tqdm(train_dataloader):
            optimizer.zero_grad()
            outputs = model(images.to(device))
            cost = loss(outputs, labels.to(device))
            cost.backward()
            optimizer.step()
        print(f"Epoch: {epoch+1}, Training Loss: {cost.item()}")

        # # Validate the Model
        # cost = 0
        # for images, labels in tqdm(val_dataloader):
        #     images.to(device)
        #     labels.to(device)
        #     outputs = model(images)
        #     cost += loss(outputs, labels).item()
        # print(f"Average Validation Loss: {cost/len(val_dataloader)}")

    # Test the Model
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in tqdm(test_dataloader):
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            print(labels)
            print(predicted)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
        print(f"Accuracy: {100 * correct / total}%")