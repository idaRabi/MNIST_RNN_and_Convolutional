import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.autograd import Variable
import matplotlib.pyplot as plt


root = './data'
download = True
learning_rate = 0.01

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5), (0.5, 0.5))])
train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
test_set = dset.MNIST(root=root, train=False, transform=trans)

batch_size = 100

train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=2)

test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=2)

# Hyper Parameters
sequence_length = 28
input_size = 28
hidden_size = 28
num_layers = 1
num_classes = 10
batch_size = 100
num_epochs = 100
learning_rate = 0.001


# RNN Model (Many-to-One)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm2_1 = nn.LSTM(input_size, hidden_size * 2, num_layers, batch_first=True, bidirectional=True)
        self.lstm2_2 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.lstm2_3 = nn.LSTM(input_size, hidden_size, 2, batch_first=True, bidirectional=True)
        self.lstm2_4 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.5)

        self.rnn2_5_1 = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.rnn2_5_2 = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.rnn2_5_3 = nn.RNN(input_size, hidden_size, 2, batch_first=True, bidirectional=True)
        self.rnn2_5_4 = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.5)

        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.fc2_2 = nn.Linear(hidden_size, num_classes)
        self.fc2_3 = nn.Linear(hidden_size * 2, num_classes)
        self.fc2_5_1 = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        return self.forward2_5_4(x)

    def forward2_1(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size))

        # Forward propagate RNN
        out, _ = self.lstm2_1(x, (h0, c0))

        # Decode hidden state of last time step
        out = F.softmax(self.fc(out[:, -1, :]), dim=1)
        return out

    def forward2_2(self, x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        out, _ = self.lstm2_2(x, (h0, c0))

        out = F.softmax(self.fc2_2(out[:, -1, :]), dim=1)

        return out

    def forward2_3(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers * 4, x.size(0), self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers * 4, x.size(0), self.hidden_size))

        # Forward propagate RNN
        out, _ = self.lstm2_3(x, (h0, c0))

        # Decode hidden state of last time step
        out = F.softmax(self.fc2_3(out[:, -1, :]), dim=1)
        return out

    def forward2_4(self, x):
        h0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size))

        out, _ = self.lstm2_4(x, (h0, c0))

        out = F.softmax(self.fc(out[:, -1, :]), dim=1)
        return out


    def forward2_5_1(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size))


        # Forward propagate RNN
        out, _ = self.rnn2_5_1(x, h0)

        # Decode hidden state of last time step
        out = F.softmax(self.fc2_5_1(out[:, -1, :]), dim=1)
        return out

    def forward2_5_2(self, x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        out, _ = self.rnn2_5_2(x, h0)

        out = F.softmax(self.fc2_2(out[:, -1, :]), dim=1)

        return out

    def forward2_5_3(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers * 4, x.size(0), self.hidden_size))

        # Forward propagate RNN
        out, _ = self.rnn2_5_3(x, h0)

        # Decode hidden state of last time step
        out = F.softmax(self.fc(out[:, -1, :]), dim=1)
        return out

    def forward2_5_4(self, x):
        h0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size))

        out, _ = self.rnn2_5_4(x, h0)

        out = F.softmax(self.fc2_5_1(out[:, -1, :]), dim=1)
        return out



rnn = RNN(input_size, hidden_size, num_layers, num_classes)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)


epochs = []
testErr = []
trainErr = []
for epoch in range(num_epochs):
    # Train the Model
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, sequence_length, input_size)
        images = Variable(images)
        labels = Variable(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = rnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_set) // batch_size, loss.data[0]))

    # Train Error
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = Variable(images.view(-1, sequence_length, input_size))
        outputs = rnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    epochTrainError = 100 - 100 * correct / total


    epochs.append(epoch)
    trainErr.append(epochTrainError)
    print("train error is %d" % epochTrainError)

    # Test Error
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, sequence_length, input_size))
        outputs = rnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    epochTestError = 100 - 100 * correct / total
    testErr.append(epochTestError)
    print("test error is %d" % epochTestError)

epochs = list(map(str, epochs))
epochs = list(map(str, epochs))
epochsTemp = []
for i in range(len(epochs)):
    if i % 2 == 0:
        epochsTemp.append(epochs[i])
    else:
        epochsTemp.append("")
epochs = epochsTemp

plt.xticks(range(len(epochs)), epochs)
plt.plot(testErr)
plt.plot(trainErr)
plt.legend(["test error", "train error"], loc="best")
plt.xlabel("epochs")
plt.ylabel("error")
plt.show()

print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))