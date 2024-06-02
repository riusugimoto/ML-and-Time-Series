import torch as t
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def get_mnist(subset: int = 1):
    '''Returns MNIST training data, sampled by the frequency given in `subset`.'''
    mnist_trainset = datasets.MNIST(root="./data", train=True, download=True, transform=MNIST_TRANSFORM)
    mnist_testset = datasets.MNIST(root="./data", train=False, download=True, transform=MNIST_TRANSFORM)

    if subset > 1:
        mnist_trainset = Subset(mnist_trainset, indices=range(0, len(mnist_trainset), subset))
        mnist_testset = Subset(mnist_testset, indices=range(0, len(mnist_testset), subset))

    return mnist_trainset, mnist_testset

def get_fashion_mnist(subset):
    # Define the transformation to convert images to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Download and load the training data
    trainset = datasets.FashionMNIST('./data', download=True, train=True, transform=transform)
    testset = datasets.FashionMNIST('./data', download=True, train=False, transform=transform)
    if subset > 1:
        trainset = Subset(trainset, indices=range(0, len(trainset), subset))
        testset = Subset(testset, indices=range(0, len(testset), subset))

    return trainset, testset


t,t = get_fashion_mnist(10)
