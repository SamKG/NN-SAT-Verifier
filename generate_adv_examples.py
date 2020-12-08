import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import pickle as pkl
import numpy as np
import analysis.verification as verif
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        assert x.shape[1:] == torch.Size(img_size)
        b = x.shape[0]
        x = x.view(1, img_size[0] * img_size[1] * img_size[2])
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

def strToFloat(str):
    if str[-1] == '?':
        str = str[:-1]
    return float(str)

def check(weight_path, delta_one, delta_digit):
    with open(weight_path, "rb") as file:
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
        def save(isSat, model, save_path):
            if isSat:
                print("is sat!")
                img = np.reshape(model, (img_size[0], img_size[1], img_size[2]))
                img = np.transpose(img, (1, 2, 0))
                fig = plt.figure()
                plt.tight_layout()
                plt.imshow(img, cmap='gray', interpolation='none')
                plt.title("Ground Truth: {}".format(y))
                plt.xticks([])
                plt.yticks([])
                plt.savefig(save_path)
            else:
                print("is not sat!")

        s = verif.RobustnessChecker([l1_weight, l2_weight], [l1_bias, l2_bias])
        for i in range(100):
            x, y = dataset.__getitem__(i)
            ground_truth = test_net.forward(x.view(1, img_size[0], img_size[1], img_size[2]))
            x = x.view(img_size[0] * img_size[1] * img_size[2]).numpy()
            s.testCorrectness(x, ground_truth.detach().view(10).numpy())

            _y = ground_truth.argmax().item()
            if y != _y:
                save(isSat=True, model=x,
                     save_path=f"adv_examples/{dataset_name}_wrong_{weight_path.replace('weights/', '')}_{i}.png")
                continue

            if delta_one is not None:
                # run single pixel
                isSat, model = s.testAllOnePixelInputRobustness(x, y, delta=delta_one)
                save(isSat, model,
                     f"adv_examples/{dataset_name}_one_{weight_path.replace('weights/', '')}_{delta_one}_{i}.png")

            if delta_digit is not None:
                # run digit
                isSat, model = s.testInputRobustness(x, y, delta=delta_digit)
                save(isSat, model,
                     f"adv_examples/{dataset_name}_digit_{weight_path.replace('weights/', '')}_{delta_digit}_{i}.png")

if __name__ == "__main__":
    dataset_name ='SVHN'
    if dataset_name == 'MNIST':
        dataset = torchvision.datasets.MNIST(
            './files/', train=True, download=True,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize(
                     (0.1307,), (0.3081,))
                 ]
            ))
        img_size = [1, 28, 28]
    elif dataset_name == 'SVHN':
        dataset = torchvision.datasets.SVHN(
                './files/', split='train', download=True,
                transform=torchvision.transforms.Compose(
                    [torchvision.transforms.Resize((16, 16)),
                     torchvision.transforms.ToTensor(),
                     torchvision.transforms.Normalize((0.4519, ), (0.1919 )),
                     ]
                )
        )
        img_size = [3, 16, 16]
    else:
            assert False

    check(weight_path=f"weights/SVHN_15_40_0.0001.pkl", delta_one=0.5,
          delta_digit=None)

# %%
