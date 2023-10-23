import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from PIL import Image

transform = transforms.Compose(
	[transforms.ToTensor(),
	transforms.Resize(256,antialias=True),
	transforms.CenterCrop(256)])

trainset = torchvision.datasets.ImageFolder(root="./training_images", transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

print('Training set has {} instances'.format(len(trainset)))

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(59536, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(x.size(0), -1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def num_flat_features(self, x):
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


#TRAINING
for epoch in range(30):
	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		inputs, labels, = data

		optimizer.zero_grad() #reset gradient on each iteration

		outputs = net(inputs) #raw output

		loss = criterion(outputs, labels) #calculate loss

		loss.backward() #calculate gradient

		optimizer.step() #optimizer is what does the actual learning, using the gradient

		running_loss += loss.item()

		if i % 10 == 9:
			print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
			running_loss = 0.0

print('Finished training')

torch.save(net.state_dict(), "./models/new_model")