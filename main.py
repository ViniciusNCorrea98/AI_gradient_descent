#Linear Regression Machine Learing Code

import torch
import time
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torch.autograd import Variable

x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

x = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

plt.plot(x_numpy, y_numpy, 'ro')
plt.show()

xn_rows, n_features = x.shape

#Number of Independent variables
input_size = n_features

output_size = 1
model = nn.Linear(input_size, output_size)

learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(model.parameters())

num_epochs = 10
conter_coust = []
for epoch in range(num_epochs):
    y_hat = model(x)
    loss = criterion(y_hat, y)
    conter_coust.append(loss)

    loss.backward()

    optimizer.step()

    if (epoch+1)%10 ==0:
        print('Epoch: ', epoch)
        print('Coust: {:.20f}'.format(loss.item()))
        print('Coefficients: ')
        print('m: {:.20f}'.format(model.weight.data.detach().item()))
        print('m (gradient): {:.20f}'.format(model.weight.grad.detach().item()))
        print('b: {:.20f}'.format(model.bias.data.detach().item()))
        print('b (gradient): {:.20f}'.format(model.bias.grad.detach().item()))

        final_preview = y_hat.detach().numpy()
        plt.plot(x_numpy, y_numpy, 'ro')
        plt.plot(x_numpy, final_preview, 'b')

        plt.show()

        time.sleep(2)
        plt.close()


    optimizer.zero_grad()

print(conter_coust)
print('Graph of Coust function')
conter_coust_np = [t.detach().numpy() for t in conter_coust]
plt.plot(conter_coust_np, 'b')
plt.show()