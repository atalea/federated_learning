import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import random
import copy

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


def train(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
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
    avg_loss = running_loss / len(train_loader)
    return accuracy, avg_loss

# Function for model aggregation


def aggregate_models(global_model, local_models):
    for global_param, local_params in zip(global_model.parameters(), zip(*[local_model.parameters() for local_model in local_models])):
        global_param.data = torch.mean(torch.stack(
            [local_param.data for local_param in local_params]), dim=0)

# Simulate federated learning using the MNIST dataset


def federated_learning(num_users, num_epochs, learning_rate, batch_size):
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
    eval_dataset = datasets.MNIST(
        root='./data', train=False, transform=transform, download=True)
    eval_loader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False)

    # Perform federated learning
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        local_models = []
        for user_id in range(num_users):
            # Create a local model
            local_model = copy.deepcopy(global_model)

            # Get the local client data
            train_indices = torch.where(train_dataset.targets == user_id)[0]

            # Check if the selected user has enough samples
            if len(train_indices) >= batch_size:
                train_data = Subset(train_dataset, train_indices)
                train_loader = DataLoader(
                    train_data, batch_size=batch_size, shuffle=True)

                # Train the local model
                accuracy, loss = train(
                    local_model, train_loader, criterion, optimizer)
                print(
                    f"User {user_id}: Loss = {loss:.4f}, Accuracy = {accuracy:.2f}%")

                # Add the local model to the list
                local_models.append(local_model)

        # Aggregate local models to update the global model
        aggregate_models(global_model, local_models)

        # Evaluate the global model
        global_model.eval()
        eval_correct = 0
        eval_total = 0
        with torch.no_grad():
            for eval_inputs, eval_labels in eval_loader:
                eval_outputs = global_model(eval_inputs)
                _, eval_predicted = torch.max(eval_outputs.data, 1)
                eval_total += eval_labels.size(0)
                eval_correct += (eval_predicted == eval_labels).sum().item()

        eval_accuracy = 100.0 * eval_correct / eval_total
        print(f"Global Model Evaluation: Accuracy = {eval_accuracy:.2f}%")

    return global_model


# Set the parameters for federated learning
num_users = 10
num_epochs = 5
learning_rate = 0.01
batch_size = 10

# Run federated learning
global_model = federated_learning(
    num_users, num_epochs, learning_rate, batch_size)
