import pickle
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import imageio
from skimage.color import rgb2gray
from skimage.transform import resize
from model import CNN
from PIL import Image
import numpy as np
input = []
for i in range(0,10000):
    filename = "./hw4_test/"
    filename = filename+str(i)+".png"
    img = imageio.imread(filename)
    # img = resize(img,(64,64),mode='constant',anti_aliasing=True)
    # img = rgb2gray(img)
    img = [[img]]
    img = torch.tensor(img)
    img = img.type(torch.FloatTensor)
    img = (img-0.5)/0.5
    # another way to process
    # img = Image.open(filename)
    # img = img.getdata()
    # data = np.array(img)
    # img = Image.reshape((64,64))

    input.append(img)
with open('processed_data_v2.txt','wb') as fp:
	pickle.dump(input,fp)