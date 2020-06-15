import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import time

root = './data'
download = True
learning_rate = 0.001

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5), (0.5, 0.5))])
train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
test_set = dset.MNIST(root=root, train=False, transform=trans)

batch_size = 10  # todo andazeie dorost o be dast biar

train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=2)

test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=2)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        conv1KS = 3
        conv2Ks = 3
        self.conv1 = nn.Conv2d(1, 7, kernel_size=conv1KS, stride=1, padding=0)
        self.conv2 = nn.Conv2d(7, 9, kernel_size=conv2Ks, stride=1, padding=0)
        self.conv1_6 = nn.Conv2d(1, 9, kernel_size=5, stride=1, padding=0)
        #soale 1.5.1
        self.conv_bn = nn.BatchNorm2d(7)
        #soale 1.5.2
        self.conv_bn2 = nn.BatchNorm2d(9)

        #soale 1.4
        self.conv2_drop = nn.Dropout2d(p=0.5)

        out1 = 28 - conv1KS + 1
        out2 = out1 - conv2Ks + 1

        # soale 1.3
        self.poolSize = 2

        self.out3 = out2 // self.poolSize
        self.fc2 = nn.Linear(self.out3 * self.out3 * 9, 50)
        self.fc1_6 = nn.Linear(24 * 24 * 9, 50)
        self.fc = nn.Linear(50, 10)

    def forward(self, x):
        #soale 1.6
        #return self.forward_1_6(x)


        x = self.conv1(x)

        # soale 1.5.1
        #x = self.conv_bn(x)

        x = self.conv2(x)

        #soale 1.5.2
        #x = self.conv_bn2(x)

        x = F.max_pool2d(F.relu(x), kernel_size=self.poolSize, stride=self.poolSize)

        ## soale 1.2
        #x = F.avg_pool2d(F.relu(self.conv2(x)), kernel_size=self.poolSize, stride=2)

        x = x.view(-1, self.out3 * self.out3 * 9)
        x = F.dropout(x, 0.2)
        x = F.relu(self.fc2(x))
        x = self.fc(x)
        return F.softmax(x, dim=1)

    def forward_1_6(self, x):
        x = self.conv1_6(x)
        x = x.view(-1, 24 * 24 * 9)
        x = F.dropout(x, 0.2)
        x = F.relu(self.fc1_6(x))
        x = self.fc(x)
        return F.softmax(x, dim=1)


model = Net()

#optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
#optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


def epoch():
    ## train
    figure = None
    epochs = []
    testErr = []
    trainErr = []
    worseningNum = 0
    prevResult = None
    for epochNum in range(20):
        ## train
        running_loss = 0.0
        model.train()
        for batch_idx, (dataa, target) in enumerate(train_loader, 0):
            dataa, target = Variable(dataa), Variable(target)
            optimizer.zero_grad()
            output = model(dataa)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

            if batch_idx % 200 == 199:
                print('[%d, %5d] loss: %.3f' % (epochNum + 1, batch_idx + 1, running_loss / 200))
                running_loss = 0.0

        ## test
        correct = 0
        total = 0
        model.eval()
        for data in test_loader:
            images, labels = data
            outputs = model(Variable(images))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        epochTestError = 100 - 100 * correct / total

        epochs.append(epochNum)
        testErr.append(epochTestError)
        print("test error is %d" % epochTestError)

        correct = 0
        total = 0
        for data in train_loader:
            images, labels = data
            outputs = model(Variable(images))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        epochTrainError = 100 - 100 * correct / total
        trainErr.append(epochTrainError)
        print("train error is %d" % epochTrainError)

        """if prevResult is None:
            prevResult = epochTestError

        if abs(epochTestError - prevResult) < 0.01:
            worseningNum += 1
        else:
            worseningNum = 0

        prevResult = epochTrainError - epochTestError

        print("worsening is %d" % worseningNum)
        if worseningNum > 5:
            break"""

    epochs = list(map(str, epochs))
    plt.xticks(range(len(epochs)), epochs)
    plt.plot(testErr)
    plt.plot(trainErr)
    plt.legend(["test error", "train error"], loc="best")
    plt.xlabel("epochs")
    plt.ylabel("error")
    plt.show()

"""def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


dataiter = iter(test_loader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print("Ground truth: ", " ".join("%5s" % labels[j] for j in range(4)))
"""

start_time = time.time()
epoch()
end_time = time.time()

print("running time is %d" % (end_time - start_time))
