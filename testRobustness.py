# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import pickle as pkl
import numpy as np
import analysis.verification as verif
import matplotlib.pyplot as plt
 
train_dataset = torchvision.datasets.MNIST('./files/', train=True, download=True,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(
                                                   (0.1307,), (0.3081,))
                                           ]))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        assert x.shape[1:] == torch.Size((1, 28, 28))
        b = x.shape[0]
        x = x.view(1, 28 * 28)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def strToFloat(str):
    if str[-1] == '?':
        str = str[:-1]
    return float(str)


with open("weights_old.pkl", "rb") as file:
    weight = pkl.load(file)
    l1_weight = weight['fc1.weight']
    l1_bias = weight['fc1.bias']
    l2_weight = weight['fc2.weight']
    l2_bias = weight['fc2.bias']
    test_net = Net()
    test_net.fc1.weight = nn.Parameter(torch.tensor(l1_weight))
    test_net.fc1.bias = nn.Parameter(torch.tensor(l1_bias))
    test_net.fc2.weight = nn.Parameter(torch.tensor(l2_weight))
    test_net.fc2.bias = nn.Parameter(torch.tensor(l2_bias))
    x, y = train_dataset.__getitem__(0)
    ground_truth = test_net.forward(x.view(1, 1, 28, 28))

    x = x.view(28*28).numpy()
    s = verif.RobustnessChecker([l1_weight, l2_weight], [l1_bias, l2_bias])
    s.testCorrectness(x, ground_truth.detach().view(10).numpy())

    print("running solver...", x, y)

    # run single pixel
    isSat, model = s.testOnePixelInputRobustness(x, y, delta=100)

    # run digit
    isSat, model = s.testInputRobustness(x, y)

    if isSat:
        print("is sat!")
        img = np.reshape(model, (28, 28))
        fig = plt.figure()
        plt.tight_layout()
        plt.imshow(img, cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(y))
        plt.xticks([])
        plt.yticks([])
        plt.savefig("adversarial.png")
    else:
        print("is not sat!")

# %%
