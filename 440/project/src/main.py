import torch as t
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import get_mnist  # Make sure to have this function properly set up for fetching data
from torch.utils.tensorboard import SummaryWriter
from model import MLP

# Import custom optimizers
from optimizers import SGD, Adam, AdaGrad, RMSProp

# Device configuration
device = t.device("cuda" if t.cuda.is_available() else "cpu")

# CNN model definition
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    

criterion = nn.CrossEntropyLoss()


# Training function
def train(model, optimizer, train_loader, test_loader, writer, epoch_index, run_number):
    model.train()
    global_step = epoch_index * len(train_loader) 
    l = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        l+= loss.item()
        
        # Log to TensorBoard
        writer.add_scalar("Loss/train", loss.item(), global_step)
        global_step +=1
    writer.add_scalar("Loss/TotalTrainLossEpoch", l, epoch_index)
    model.eval()
    correct = 0
    total = 0
    with t.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = t.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    writer.add_scalar("Accuracy/test", accuracy, epoch_index)
    print(f'Epoch [{epoch_index+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

# Hyperparameters
optimizers_params = [
    {'name': 'SGD', 'lr': 0.01, 'momentum': 0.0},
    {'name': 'SGD', 'lr': 0.001, 'momentum': 0.0},
    {'name': 'SGD', 'lr': 0.01, 'momentum': 0.9},
    {'name': 'SGD', 'lr': 0.001, 'momentum': 0.95},
    {'name': 'Adam', 'lr': 0.01, 'beta1': 0.9, 'beta2': 0.999},
    {'name': 'Adam', 'lr': 0.001, 'beta1': 0.95, 'beta2': 0.995},
    {'name': 'RMSProp', 'lr': 0.01, 'alpha': 0.99},
    {'name': 'RMSProp', 'lr': 0.001, 'alpha': 0.95},
    {'name': 'Adagrad', 'lr': 0.01},
    {'name': 'Adagrad', 'lr': 0.001}
]
batch_sizes = [2, 16,32, 64]
many_runs = 2
epochs = 10
subset = 30

inp = 784
hid = 30
out = 10

train_data, test_data = get_mnist(subset)

for opt_params in optimizers_params:
    for batch_size in batch_sizes:
        for run_number in range(1,many_runs+1):
            print(f"Training run number {run_number} with {opt_params['name']} at LR={opt_params['lr']} and batch size={batch_size}")

            # DataLoader setup
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

            model = MLP(input=inp, hidden=hid, output=out).to(device)


            # Optimizer initialization
            if opt_params['name'] == 'SGD':
                optimizer = SGD(model.parameters(), lr=opt_params['lr'], momentum=opt_params['momentum'])
                writer = SummaryWriter(f"mnist/MLP_{opt_params['name']}_LR{opt_params['lr']}_Momentum{opt_params['momentum']}_BS{batch_size}_TEST")
            elif opt_params['name'] == 'Adam':
                optimizer = Adam(model.parameters(), lr=opt_params['lr'], b1=opt_params['beta1'], b2=opt_params['beta2'])
                writer = SummaryWriter(f"mnist/MLP_{opt_params['name']}_LR{opt_params['lr']}_B1{opt_params['beta1']}_B2{opt_params['beta2']}_BS{batch_size}")

            elif opt_params['name'] == 'RMSProp':
                optimizer = RMSProp(model.parameters(), lr=opt_params['lr'], alpha=opt_params['alpha'])
                writer = SummaryWriter(f"mnist/MLP_{opt_params['name']}_LR{opt_params['alpha']}_BS{batch_size}")
            elif opt_params['name'] == 'Adagrad':
                optimizer = AdaGrad(model.parameters(), lr=opt_params['lr'])
                writer = SummaryWriter(f"mnist/MLP_{opt_params['name']}_LR{opt_params['lr']}_BS{batch_size}")

            
            for epoch in range(epochs): 
                train(model, optimizer, train_loader, test_loader, writer, epoch, run_number)

            writer.close()

print("Grid search completed.")
