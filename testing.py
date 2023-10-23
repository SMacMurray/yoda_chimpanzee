import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


transform = transforms.Compose(
	[transforms.ToTensor(),
	transforms.Resize(256,antialias=True),
	transforms.CenterCrop(256)])

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
net.load_state_dict(torch.load("./models/new_model"))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


classes = torchvision.datasets.ImageFolder("./training_images", transform=transform).classes
print(f"Classes: {classes}")



# testing using directory of test images
with torch.no_grad():
    for i in range(len(os.listdir("./testing_images"))):
        img_path = "./testing_images/" + str(i) + ".jpg"
        image = Image.open(img_path)
        image = transform(image).float()
        image = image.unsqueeze(0)

        net.eval()
        output = net.forward(image)
        index = output.data.argmax().item()

        prediction = net(image).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        
        print(f"{i}:{classes[index]}: with {score*100:.1f}% confidence.")



'''
# test using a single image, and show it
img_path = "./testing_images/0.jpg"
image = Image.open(img_path)
image = transform(image).float()
image = image.unsqueeze(0)

with torch.no_grad():
    image = Image.open(img_path)
    image = transform(image).float()
    image = image.unsqueeze(0)

    net.eval()
    output = net.forward(image)
    index = output.data.argmax().item()

    prediction = net(image).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    
    print(f"{classes[index]}: with {score*100:.1f}% confidence.")

imgplot = plt.imshow(mpimg.imread(img_path))
plt.show()
'''