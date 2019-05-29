import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from model import *
import torchvision.models as models
import copy
# Hyper Parameters
num_epochs = 30
batch_size = 128
learning_rate = 0.0001
#preprocessing Dataset
data_transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        # transforms.Resize(64),
        # transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5],
                             std=[0.5])
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

# CNN Model (2 conv layer)
# model = CNN()
# model = FashionSimpleNet()
model = models.resnet18()
# model = models.alexnet()
model.cuda()
# If you want to finetune only top layer of the model.
# for param in resnet.parameters():
#     param.requires_grad = False
    
# # Replace top layer for finetuning.
# resnet.fc = nn.Linear(resnet.fc.in_features, 100)  # 100 is for example.
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adam(model.parameters())
# Train the Model
best_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1,num_epochs))
    print('-'*10)
    for phase in [0, 1]:
        if phase == 0:
            model.train()
        else:
            model.eval()
        running_loss = 0.0
        running_corrects = 0
        for (images, labels) in (data_loader[phase]):
            images = Variable(images).cuda()
            labels = Variable(labels).cuda() 
            # images = Variable(images)
            # labels = Variable(labels)    
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase==0):
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs,1)
                if phase==0:
                    loss.backward()
                    optimizer.step()
    #statistics
        running_loss+=loss.item()
        running_corrects+=torch.sum(preds==labels.data)
    # epoch_loss = running_loss/len(data_loader[phase].dataset)
    # epoch_acc = running_corrects.double()/len(data_loader[phase].dataset)
        run_loss = running_loss/len(data_loader[phase])
        epoch_loss = running_loss/len(data_loader[phase].dataset)
        epoch_acc = running_corrects.double()/len(data_loader[phase].dataset)
        if(phase==0):
            print('training loss: {:.4f} running_Loss: {:.4f} Acc:{:.4f}'.format(run_loss,epoch_loss,epoch_acc))
        else:
            print('validation loss: {:.4f} running_Loss: {:.4f} Acc:{:.4f}'.format(run_loss,epoch_loss,epoch_acc))
    if phase == 1 and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
print()
        # if (i+1) % 50 == 0:
        #     print("Epoch {}/{}, Loss: {:.3f}".format(epoch+1,num_epochs, loss.item()))
        # if (i+1) %100 == 0:
        #     valid(model,valid_loader,criterion)
    # loss_train.append(train(cnn,train_loader,criterion,optimizer))
    # loss_valid.append(valid(cnn,valid_loader,criterion))
model.load_state_dict(best_model_wts)
# Save the Trained Model
torch.save(model.state_dict(), 'cnn.pkl')