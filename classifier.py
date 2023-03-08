import torch
import torchvision
import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'

mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)# make my data
x_train = mnist_train.data.reshape(-1, 784).float()  # Reshape input
# Create output tensor
y_train = torch.zeros((mnist_train.targets.shape[0], 10))
y_train[torch.arange(mnist_train.targets.shape[0]),
        mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True) # make my data
x_test = mnist_test.data.reshape(-1, 784).float()  # Reshape input
y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
y_test[torch.arange(mnist_test.targets.shape[0]),
       mnist_test.targets] = 1  # Populate output


class SoftmaxModel:
    def __init__(self):
        # Model variables
        self.W = torch.randn(784, 10, requires_grad=True)
        self.b = torch.randn(10, requires_grad=True)

    def f(self, x):
        return torch.nn.functional.softmax(x @ self.W + self.b, dim=1)

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.cross_entropy(self.f(x), y)

    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


for i in range(1000):
    model = SoftmaxModel()
    optimizer = torch.optim.Adam([model.W, model.b], 0.05)
    for epoch in range(500):
        model.loss(x_train, y_train).backward()
        optimizer.step()

        optimizer.zero_grad()
        if epoch % 100 == 0:
            print("E = %s, L = %s" % (epoch, model.loss(x_train, y_train)))

    print(model.accuracy(x_train, y_train))
    if (model.accuracy(x_test, y_test).item() > 0.9):
        # Display the weights after they have been trained
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.imshow(model.W[:, i].detach().numpy().reshape(28, 28))
            plt.title(f'img: {i}')
            plt.xticks([])
            plt.yticks([])

        plt.show()