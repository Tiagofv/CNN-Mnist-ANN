import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as fn
from os.path import join
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataloader import MNISTReader

# Set file paths based on added MNIST Datasets
input_path = './mnist'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')


# Helper function to show a list of images with their relating titles
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(30, 20))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize=15);
        index += 1
    plt.show()


# load mnist dataset
mnist_dataloader = MNISTReader(training_images_filepath, training_labels_filepath, test_images_filepath,
                               test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load()


images_2_show = []
titles_2_show = []
for i in range(0, 10):
    r = random.randint(1, 60000)
    images_2_show.append(x_train[r])
    titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))

for i in range(0, 5):
    r = random.randint(1, 10000)
    images_2_show.append(x_test[r])
    titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))

show_images(images_2_show, titles_2_show)


# basic cnn for mnist problem

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(3 * 3 * 64, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):
        x = fn.relu(self.conv1(x))
        x = fn.relu(self.conv2(x))
        x = fn.max_pool2d(x, 2)
        x = self.dropout(x)
        x = fn.relu(self.conv3(x))
        x = fn.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(-1, 3 * 3 * 64)
        x = fn.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return fn.log_softmax(x, dim=1)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = fn.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += fn.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


x_train = np.array(x_train).reshape(-1, 1, 28, 28) # ensure data has only one channel (grayscale)
x_test = np.array(x_test).reshape(-1, 1, 28, 28)

train_loader = DataLoader(TensorDataset(torch.Tensor(x_train), torch.tensor(np.array(y_train))),
                          batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.Tensor(x_test), torch.tensor(np.array(y_test))), batch_size=64,
                         shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, 3):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

# test model
model.eval()
with torch.no_grad():
    output = model(torch.Tensor(x_test[0].reshape(1, 1, 28, 28)).to(device))
    print('Prediction:', output.argmax(dim=1).item())
    print('Actual:', y_test[0])
    plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
    plt.show()


# plot confusion matrix using matplotlib

preds = []
actuals = []
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        preds += output.argmax(dim=1).tolist()
        actuals += target.tolist()

confusion_matrix = confusion_matrix(actuals, preds)
ConfusionMatrixDisplay(confusion_matrix).plot()
plt.show()

# save model
torch.save(model.state_dict(), 'mnist_cnn.pt')
