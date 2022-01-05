import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def mnist():
    # exchange with the corrupted mnist dataset
    from torchvision import datasets, transforms

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])

    path  = '/Users/christiandjurhuus/Documents/DTU/6_semester/ml_ops/dtu_mlops/data/corruptmnist/'
    # Download and load the training data
    train = np.load(path + "training_data.npz")
    images = torch.Tensor(train.f.images)
    labels = torch.Tensor(train.f.labels).type(torch.LongTensor)
    trainset = TensorDataset(images, labels)


#    num_row = 10
#    num_col = 5  # plot images
#    num = num_col * num_row
#    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
#    for i in range(num):
#        ax = axes[i // num_col, i % num_col]
#        ax.imshow(images[i], cmap='gray')
#        ax.set_title('Label: {}'.format(labels[i]))
#    plt.tight_layout()
#    plt.show()

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Download and load the test data
    testset = np.load(path + "test.npz")
    images_test = torch.Tensor(testset.f.images)
    labels_test = torch.Tensor(testset.f.labels).type(torch.LongTensor)
    testset = TensorDataset(images_test, labels_test)


    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    return trainloader, testloader


if __name__ == '__main__':
    mnist()