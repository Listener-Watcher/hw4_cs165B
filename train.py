import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
import copy
import numpy as np
import matplotlib.pyplot as plt
from model import *
# Hyper Parameters
num_epochs = 60
batch_size = 128
early_stop = 5
learning_rate = 0.01
# fix random seed

torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
np.random.seed(0)
#preprocessing Dataset
def adjust_learning_rate(optimizer, epoch):
    global learning_rate
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    learning_rate = learning_rate * (0.1 ** (epoch // 8))
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


data_transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0,5,0,5],
                             std=[0.5,0,5,0,5])
    ])
train_dataset = dsets.ImageFolder(root='./hw4_train',transform=data_transform)
train_size = int(0.8 * len(train_dataset))
test_size = len(train_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])
data_loader = []

train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size, shuffle=True
                                             )
valid_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size, shuffle=True
                                             )
data_loader.append(train_loader)
data_loader.append(valid_loader)
# train_iter = iter(train_loader)
# print(type(train_iter))
# image,label = train_iter.next()
# print('images shape on batch size = {}'.format(image.size()))
# print('labels shape on batch size = {}'.format(label.size()))
def save_checkpoint(model, optimizer, epoch, filename):
    state = {"model_state_dict": model.state_dict(),
             "optimizer": optimizer.state_dict(),
             "epoch": epoch}
    
    torch.save(state, filename)

def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.cuda()

    optimizer.load_state_dict(checkpoint["optimizer"])
    for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

# CNN Model (2 conv layer)
# model = CNN()
# model = FashionSimpleNet()
#model = models.resnet52()
#model.fc = nn.Linear(model.fc.in_features, 10)
model = CNN2()
model.cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adam(model.parameters())
# Train the Model
best_acc = 0.0
val_loss =[]
train_loss = []
epoch_count = []
best_model_wts = copy.deepcopy(model.state_dict())
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1,num_epochs))
    print('-'*10)
    epoch_count.append(epoch)
    # adjust_learning_rate(optimizer, epoch)
    for phase in [0, 1]:
        if phase == 0:
            model.train()
        else:
            model.eval()
        running_loss = 0.0
        label_corrects = [1]*10
        label_count = [1]*10
        running_corrects = 0
        size = 0
        for (images, labels) in (data_loader[phase]):
            images = Variable(images).cuda()
            labels = Variable(labels).cuda() 
            # images = Variable(images)
            # labels = Variable(labels)    
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            size +=1
            with torch.set_grad_enabled(phase==0):
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs,1)
                if phase==0:
                    loss.backward()
                    optimizer.step()
                if phase==1:
                    label_corrects[labels]+=(preds==labels.data)
                    label_count[labels]+=1
    #statistics
        running_loss+=loss.item()
        running_corrects=torch.sum(preds==labels.data)
        for i in range(0,10):
            ave_corrects += (label_corrects[i].double()/label_count[i])
        ave_corrects = ave_corrects/10
    # epoch_loss = running_loss/len(data_loader[phase].dataset)
    # epoch_acc = running_corrects.double()/len(data_loader[phase].dataset)
        run_loss = running_loss
        # epoch_loss = running_loss/len(data_loader[phase].dataset)
        epoch_acc = running_corrects.double()
        # acc = np.mean(running_corrects)
        if(phase==0):
            train_loss.append(run_loss)
        else:
            val_loss.append(run_loss)
        if(phase==0):
            print('training loss: {:.8f} epoch_acc: {:.8f}'.format(run_loss/train_size,epoch_acc/train_size))
        else:
            print('validation loss: {:.8f} epoch_acc: {:.8f} ave_corrects:{:.8f}'.format(run_loss/test_size,epoch_acc/test_size,ave_corrects))
    if epoch_acc >= best_acc:
        early_stop = 5
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        # save_checkpoint(model, optimizer, epoch, "save_model_"+str(epoch))
    else:
        early_stop -= 1
        if(early_stop==0):
            save_checkpoint(model, optimizer, epoch, "cnn2_save_model_"+str(epoch))

print()
plt.plot(epoch_count,train_loss,c="b")
plt.show()
plt.plot(epoch_count,val_loss,c="r")
plt.show()
model.load_state_dict(best_model_wts)
# Save the Trained Model
torch.save(model.state_dict(), 'cnn2_'+str(num_epochs)+'.pkl')
