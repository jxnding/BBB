"""
Adapt from:
1. https://gist.github.com/vvanirudh/9e30b2f908e801da1bd789f4ce3e7aac
2. "Variational Dropout and the Local Reparameterization Trick" (https://arxiv.org/abs/1506.02557)
3. http://gluon.mxnet.io/chapter18_variational-methods-and-uncertainty/bayes-by-backprop.html
"""
from BNNLayer import BNNLayer
from BNN import BNN

import torch
import torchvision
import torch.utils.data as Data
from torch.autograd import Variable

import pdb
import numpy as np
import sys
import math

def load_images():
    images = np.load(sys.argv[1],allow_pickle=True)
    train, test = [], []
    for c in ('[0]','[1]','[2]','[3]','[4]'):
        curr_class = images.item().get(c)
        c = int(c[1])
        for i in range(1000):
            image = curr_class[i]
            image = np.reshape(image, (1, 4))
            train.append([image, c])
        for i in range(1000,1200):
            image = curr_class[i]
            image = np.reshape(image, (1, 4))
            test.append([image, c])
        # for image in curr_class:
        #     image_set.append([image[0], c])
    return train, test

N_Epochs = 50
N_Samples = 10
LearningRate = 1e-2
Download_MNIST = True   # download the dataset if you don't already have it

# Change to whatever directory your data is at
import os.path
dataset_path = os.path.join(os.path.dirname(__file__), 'mnist')

# train_set = torchvision.datasets.MNIST(
#     root=dataset_path,
#     train=True,
#     transform=torchvision.transforms.ToTensor(),
#     download=Download_MNIST
# )

N_Batch = 1

# train_loader = Data.DataLoader(dataset=train_set, batch_size=BatchSize, shuffle=True)
train_images, test_images = load_images()
# pdb.set_trace()
# test_set = torchvision.datasets.MNIST(
#     root='./mnist/',
#     train=False,
#     transform=torchvision.transforms.ToTensor(),
#     download=Download_MNIST
# )

# test_size = test_set.test_data.size()[0]

compute_accu = lambda pred, true, digits: round((pred == true).mean() * 100, digits)

if __name__ == '__main__':

    # Initialize network
    bnn = BNN(BNNLayer(4, 5, activation='softmax', prior_mean=math.exp(-0), prior_rho=math.exp(-6)))
    optim = torch.optim.Adam(bnn.parameters(), lr=LearningRate)

    # Main training loop
    train_accu_lst = []
    test_accu_lst = []
    w_mean = []
    w_rho = []
    for i_ep in range(N_Epochs):

        # Training
        for X, Y in train_images:
            X = torch.Tensor(X)
            # X, Y = torch.from_numpy(X), torch.from_numpy(Y)
            kl, log_likelihood = bnn.Forward(X, Y, N_Samples, type='Softmax')

            # Loss and backprop
            loss = BNN.loss_fn(kl, log_likelihood, N_Batch)
            optim.zero_grad()
            loss.backward()
            optim.step()

        # Evaluate on training set
        train_X = [x for x,y in train_images]
        train_Y = [y for x,y in train_images]
        # pdb.set_trace()

        if i_ep in [2, 5, 10]:
            w_mean.append(bnn.layers[0].W_mean.clone())
            w_rho.append(bnn.layers[0].W_rho.clone())

        pred_class = bnn.forward(torch.Tensor(np.concatenate(train_X)), mode='MAP').data.numpy().argmax(axis=1)
        # pred_class = bnn.forward(torch.Tensor(train_X), mode='MAP').data.numpy().argmax(axis=1)
        true_class = train_Y

        train_accu = compute_accu(pred_class, true_class, 1)
        print('Epoch', i_ep, '|  Training Accuracy:', train_accu, '%')

        train_accu_lst.append(train_accu)

        # # Evaluate on test set
        test_X = [x for x,y in test_images]
        test_Y = [y for x,y in test_images]

        pred_class = bnn.forward(torch.Tensor(np.concatenate(test_X)), mode='MAP').data.numpy().argmax(axis=1)
        true_class = test_Y

        test_accu = compute_accu(pred_class, true_class, 1)
        print('Epoch', i_ep, '|  Test Accuracy:', test_accu, '%')

        test_accu_lst.append(test_accu)

    # Plot
    import matplotlib.pyplot as plt
    iters = [2, 5, 10]
    # pdb.set_trace()
    title = 'Var 0.3 '
    for i in range(3):
        plt.title('Small Network '+str(iters[i]))
        print(np.mean(w_mean[i].detach().numpy().reshape(20)))
        plt.errorbar([x for x in range(20)], w_mean[i].detach().numpy().reshape(20), w_rho[i].detach().numpy().reshape(20), fmt='o-')
        # plt.savefig('Small Network '+str(iters[i])+'.png',dpi=150)
        plt.xlabel('Weight Number')
        plt.ylabel('Mean')
        plt.title('Learned Weights, '+title)
        # plt.legend(['BNN, Acc: %2.2f' % test_accu_lst[iters[i]]/100 ])

        plt.savefig('Learned Weights, '+title+str(iters[i])+'.png', dpi=150)
        plt.close()
    # plt.style.use('seaborn-paper')

    # plt.title('Classification Accuracy on MNIST')
    # plt.plot(train_accu_lst, label='Train')
    # plt.plot(test_accu_lst, label='Test')
    # plt.ylabel('Accuracy (%)')
    # plt.xlabel('Epochs')
    # plt.legend(loc='best')
    # plt.tight_layout()
    # plt.show()
