import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import random
import heapq
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Define the model architecture


class CNNCifar(nn.Module):
    def __init__(self):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# Function to train the model on local data and calculate accuracy


def train(model, train_loader, optimizer, criterion, local_epochs):
    model.train()
    for epoch in range(local_epochs):
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

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        accuracy = 100.0 * correct / total
        # print(f"Epoch {epoch+1}: Loss = {running_loss/len(train_loader):0.2f}, Accuracy = {accuracy:0.2f}%")
        return accuracy, float(loss)

# Function to test the global model


def test(model, test_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation during testing
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    average_loss = running_loss / len(test_loader)
    return accuracy, average_loss
# This is FedAvg algorithm


def federated_learningFedAvg(local_epochs, global_epochs,  learning_rate):
    temp_power = []
    # Create the global model
    global_model = CNNCifar()
    criterion = nn.CrossEntropyLoss()

    # Transformation for CIFAR-10 (RGB images)
    trans_cifar = transforms.Compose([
        transforms.ToTensor(),
        # Normalize for RGB images
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(
        root='./data/cifar10', train=True, download=True, transform=trans_cifar)

    test_dataset = datasets.CIFAR10(
        root='./data/cifar10', train=False, download=True, transform=trans_cifar)

    # Convert target labels to tensors
    train_dataset.targets = torch.tensor(train_dataset.targets)
    test_dataset.targets = torch.tensor(test_dataset.targets)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_bs, shuffle=True)

    # Split the dataset into client data
    client_data = custom_data_split(train_dataset, len(clients))

    # print(f'The random selected users for FedAvg are {clients}')

    # Perform federated learning
    print('This is FedAvg algorithm')
    for i in range(global_epochs):
        print(f'Global epoch # {i}')
        # Select active clients for fedAvg
        # print(len(clients))
        wireless_channel_transition_probability(clients)
        active = rand_active_user(clients, top_k)
        # print(f'active users {active}')
        # print(f'client status{clients_state}')
        successfull_users = []
        for i in range(len(active)):
            if clients_state[active[i]] == 0:
                # print(f'active users {i} is {active[i]}')
                successfull_users.append(clients_state[active[i]])
                temp_power.append(0)
            # calculate non-successfull users power
            else:
                temp_power.append(clients_power[i])
        # print(f'temp power {temp_power}')
        # print(f'Successfull clients for FedAvg are: {successfull_users}')

        # This is for the fedAvg training
        if (successfull_users == []):
            print('No user is able to transmit')
            # print(f'Global Accuracy {acc: .2f}, global loss {loss: .2f}')
        else:

            for i in range(len(successfull_users)):
                # print(f"This is user {i+1}, their ID is: {active[i]}")
                local_model = CNNCifar()
                local_model.load_state_dict(global_model.state_dict())

                # Get the local client data
                train_data = client_data[i]
                train_loader = torch.utils.data.DataLoader(
                    train_data, batch_size=local_bs, shuffle=True)

                # Create an optimizer for the local model
                optimizer = optim.SGD(
                    local_model.parameters(), lr=learning_rate, momentum=0.5)

                # Train the local model
                acc, loss = train(local_model, train_loader,
                                  optimizer, criterion, local_epochs)
                # print(f'train acc {acc}, train loss {loss}')

            # Update the global model with the local model's parameters
            global_model.load_state_dict(local_model.state_dict())
            # print(f'Global Accuracy {acc: .2f}, global loss {loss: .2f}')
            test_accuracy, test_loss = test(
                global_model, test_loader, criterion)
            # print(f"Testing Accuracy: {test_accuracy:.2f}%")
            # print(f"Testing Loss: {test_loss:.2f}")
            fedavg_accu.append(test_accuracy)
            fedavg_loss.append(test_loss)
            fedavg_power.append(sum(temp_power)/len(temp_power))

    return global_model


def federated_learningIBCS(local_epochs, global_epochs,  learning_rate):
    # Create the global model
    temp_power = []
    global_model = CNNCifar()
    criterion = nn.CrossEntropyLoss()

    # Transformation for CIFAR-10 (RGB images)
    trans_cifar = transforms.Compose([
        transforms.ToTensor(),
        # Normalize for RGB images
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(
        root='./data/cifar10', train=True, download=True, transform=trans_cifar)

    test_dataset = datasets.CIFAR10(
        root='./data/cifar10', train=False, download=True, transform=trans_cifar)

    # Convert target labels to tensors
    train_dataset.targets = torch.tensor(train_dataset.targets)
    test_dataset.targets = torch.tensor(test_dataset.targets)
    # print(f'test data leng is {len(test_dataset)}')
    # Convert target labels to tensors
    train_dataset.targets = torch.tensor(train_dataset.targets)
    test_dataset.targets = torch.tensor(test_dataset.targets)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_bs, shuffle=True)

    # Split the dataset into client data
    client_data = custom_data_split(train_dataset, len(clients))

    # select top K clients after indexing
    print('\n')
    print('This is IBCS algorithm')
    # Perform federated learning
    for i in range(global_epochs):
        print(f'Global epoch # {i}')
        # Select active clients for top_k indexing algorithm
        wireless_channel_transition_probability(clients)
        active = clients_indexing(clients, clients_power)
        # print(f'active users {active}')
        # print(f'active are {active}')
        # print(f'status {clients_state}')
        successfull_users = []
        for i in range(len(active)):
            if clients_state[active[i]] == 0:
                successfull_users.append(clients_state[active[i]])
                temp_power.append(0)
            else:
                temp_power.append(clients_power[active[i]])
        # print(f'temp power {temp_power}')
        # print(f'Successfull clients for ibcs are: {successfull_users}')

        # This is for the fedAvg training
        if (successfull_users == []):
            print('No user is able to transmit')
            # print(f'Global Accuracy {acc: .2f}, global loss {loss: .2f}')
        else:

            for i in range(len(successfull_users)):
                # print(f"This is user {i+1}, their ID is: {active[i]}")
                local_model = CNNCifar()
                local_model.load_state_dict(global_model.state_dict())

                # Get the local client data
                train_data = client_data[i]
                train_loader = torch.utils.data.DataLoader(
                    train_data, batch_size=local_bs, shuffle=True)

                # Create an optimizer for the local model
                optimizer = optim.SGD(
                    local_model.parameters(), lr=learning_rate, momentum=0.5)

                # Train the local model
                acc, loss = train(local_model, train_loader,
                                  optimizer, criterion, local_epochs)

            # Update the global model with the local model's parameters
            global_model.load_state_dict(local_model.state_dict())
            # print(f'Global Accuracy {acc: .2f}, global loss {loss: .2f}')
            test_accuracy, test_loss = test(
                global_model, test_loader, criterion)
            # print(f"Testing Accuracy: {test_accuracy:.2f}%")
            # print(f"Testing Loss: {test_loss:.2f}")
            ibcs_accu.append(test_accuracy)
            ibcs_loss.append(test_loss)
            ibcs_power.append(sum(temp_power)/len(temp_power))
    # print(transition_prob)
    # collect the data (power, loss, acc) --> plot()

    return global_model


# Set the parameters for federated learning
def clients_pool(num_clients):
    clients = []
    for i in range(num_clients):
        clients.append(i)
    return clients

# assign power to clients


def power(clients):
    clients_power = []
    for i in range(len(clients)):
        rand = random.randint(1, 100)
        clients_power.append(rand)
    return clients_power


#################################################################################################
num_clients = int(input('Please Enter the number of clients: '))
clients = clients_pool(num_clients)
clients_power = power(clients)
global_epochs = 10
local_epochs = 100
learning_rate = 0.01
top_k = 100
state_0 = [0.9449, 0.0087, 0.9913]
state_1 = [0.0551, 0.8509, 0.1491]
clients_state = []

local_bs = 10
test_bs = 128

fedavg_accu = []
fedavg_loss = []
fedavg_power = []
ibcs_accu = []
ibcs_loss = []
ibcs_power = []
#################################################################################################
# slect a random number of active users


def user_selection_fedAvg(clients, top_k):
    if top_k >= len(clients):
        return clients

    subset = random.sample(clients, top_k)
    return subset

# select subset from the random selected users for fedAvg


def rand_active_user(clients, top_k):
    if top_k >= len(clients):
        return clients
    active = random.sample(clients, top_k)
    # print(f'active users {len(active)}')
    return active

# select top_k users


def clients_indexing(clients, clients_power):
    user_indices = []
    for i in range(len(clients)):
        if clients_state[i] == 1:
            v_i_t = -(state_1[1]/len(clients)) - \
                (((state_1[0]*clients_power[i])/100))
            user_indices.append(v_i_t)
            # print(f'client {clients[i]}, is in state {clients_state[i]}')
        elif clients_state[i] == 0:
            v_i_t = -(state_0[1]/len(clients)) - \
                (((state_0[0]*clients_power[i])/100))
            user_indices.append(v_i_t)
    # print('Indices are', user_indices)
    # this prints the top k values
    top_k_users = heapq.nlargest(top_k, user_indices)
    # print(f'the top {top_k} users who can transmit are: {top_k_users}')
    # this prints the top k indices
    user_indices = np.argsort(user_indices)
    top_k_users = user_indices[-top_k:]
    # print(f'the top {top_k} users who can transmit are: {top_k_users}')
    # print(f'client {clients[i]}, is in state {clients_state[i]}')
    return top_k_users

# transition probability calculation


def wireless_channel_transition_probability(clients):
    temp = []
    if clients_state == []:
        # print('This is time 0')
        for i in range(len(clients)):
            # print(f'clien stae {i}')
            rand_transision = random.random()
            if rand_transision <= state_0[0]:
                # print(f'random here is {rand_transision}')
                clients_state.append(0)
            else:
                # print(f'random here is {rand_transision}')
                clients_state.append(1)
    else:
        # print('This is Not time 0')
        for i in range(len(clients)):
            rand_transision = random.random()
            # print(f'random here is {rand_transision}')
            if clients_state[i] == 0:
                if rand_transision <= state_0[1]:
                    clients_state[i] = 1
                else:
                    clients_state[i] = 0
            else:
                if rand_transision <= state_0[2]:
                    clients_state[i] = 0
                else:
                    clients_state[i] = 1

# Custom function for data splitting to ensure balanced subsets


# Custom function for data splitting to ensure balanced subsets
def custom_data_split(dataset, num_clients):
    num_samples = len(dataset)
    samples_per_client = num_samples // num_clients
    remainder = num_samples % num_clients

    client_data = []
    current_idx = 0

    for i in range(num_clients):
        client_size = samples_per_client + (1 if i < remainder else 0)
        indices = list(range(current_idx, current_idx + client_size))
        current_idx += client_size
        client_data.append(torch.utils.data.Subset(dataset, indices))

    return client_data


# Run federated learning
global_model = federated_learningFedAvg(
    local_epochs, global_epochs, learning_rate)

print(f'fedavg accuracy {fedavg_accu}')
print(f'fedavg loss {fedavg_loss}')
print(f'fedavg power {fedavg_power}')


# reset states before you start the second algorithm
clients_state = []

global_model = federated_learningIBCS(
    local_epochs, global_epochs, learning_rate)

print(f'ibcs accuracy {ibcs_accu}')
print(f'ibcs loss {ibcs_loss}')
print(f'ibcs power {ibcs_power}')


def plot():
    fig, ax = plt.subplots()
    ax.plot(fedavg_accu)
    ax.plot(ibcs_accu)

    ax.set_title('Accuracy')
    ax.legend(['FedAvg', 'IBCS'])
    ax.xaxis.set_label_text('Gobal Epochs')
    ax.yaxis.set_label_text('Accuracy in %')
    # plt.show()
    plt.savefig('./results/Accuracy_CIFAIR10.png')

    fig, ax = plt.subplots()
    ax.plot(fedavg_loss)
    ax.plot(ibcs_loss)

    ax.set_title('Loss')
    ax.legend(['FedAvg', 'IBCS'])
    ax.xaxis.set_label_text('Gobal Epochs')
    ax.yaxis.set_label_text('Loss')
    # plt.show()
    plt.savefig('./results/Loss_CIFAIR10.png')

    fig, ax = plt.subplots()
    ax.plot(fedavg_power)
    ax.plot(ibcs_power)

    ax.set_title('Power')
    ax.legend(['FedAvg', 'IBCS'])
    ax.xaxis.set_label_text('Gobal Epochs')
    ax.yaxis.set_label_text('Power')
    # plt.show()
    plt.savefig('./results/Power_CIFAIR10.png')


plot()
