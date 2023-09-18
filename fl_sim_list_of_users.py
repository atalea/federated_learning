import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import random

# Define the model architecture


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to train the model on local data and calculate accuracy and loss


def train(model, train_loader, criterion, optimizer, user_id):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / total
        print(
            f"User {user_id}: Epoch {epoch+1}/{num_epochs} Loss = {running_loss:.4f}, Accuracy = {accuracy:.2f}%")

# Function for model aggregation


def aggregate_models(global_model, local_models):
    for global_param, local_params in zip(global_model.parameters(), zip(*[local_model.parameters() for local_model in local_models])):
        global_param.data = torch.mean(torch.stack(
            [local_param.data for local_param in local_params]), dim=0)

# Simulate federated learning using the MNIST dataset


def federated_learning(user_ids, num_epochs, learning_rate, batch_size):
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    # Create the global model
    global_model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(global_model.parameters(), lr=learning_rate)

    # Load the MNIST dataset
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        root='./data', train=True, transform=transform, download=True)

    # Perform federated learning
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        local_models = []
        for user_id in user_ids:
            # Create a local model
            local_model = Net()
            local_model.load_state_dict(global_model.state_dict())

            # Get the local client data
            train_indices = torch.where(train_dataset.targets == user_id)[0]

            # Check if the selected user has enough samples
            if len(train_indices) >= batch_size:
                train_data = Subset(train_dataset, train_indices)
                train_loader = DataLoader(
                    train_data, batch_size=batch_size, shuffle=True)

                # Train the local model
                train(local_model, train_loader, criterion, optimizer, user_id)

                # Add the local model to the list
                local_models.append(local_model)

        # Aggregate local models to update the global model
        aggregate_models(global_model, local_models)

        # Evaluate the global model on each user's data
        for user_id in user_ids:
            eval_indices = torch.where(train_dataset.targets == user_id)[0]
            if len(eval_indices) >= batch_size:
                eval_data = Subset(train_dataset, eval_indices)
                eval_loader = DataLoader(
                    eval_data, batch_size=batch_size, shuffle=True)
                evaluate(global_model, eval_loader, criterion, user_id)

    return global_model

# Function to evaluate the global model on user's data


def evaluate(model, eval_loader, criterion, user_id):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in eval_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    print(
        f"User {user_id}: Evaluation Loss = {running_loss:.4f}, Accuracy = {accuracy:.2f}%")


# Set the parameters for federated learning
user_ids = [0, 1, 2, 3, 4]  # User IDs to participate in federated learning
num_epochs = 5
learning_rate = 0.01
batch_size = 10

# Run federated learning
global_model = federated_learning(
    user_ids, num_epochs, learning_rate, batch_size)
