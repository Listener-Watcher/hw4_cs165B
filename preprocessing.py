import pickle
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import imageio
from skimage.color import rgb2gray,gray2rgb
from skimage.transform import resize
from PIL import Image
import numpy as np
input = []
for i in range(0,10000):
    filename = "./hw4_test/"
    filename = filename+str(i)+".png"
    #img = imageio.imread(filename)
    #img = resize(img,(64,),mode='constant',anti_aliasing=True)
    #img = rgb2gray(img)

    # # img = gray2rgb(img)
    #img = (img-0.5)/0.5
    # # img = img/255
    #img = [[img]]
    #img = torch.tensor(img)
    #img = img.type(torch.FloatTensor)


    # another way to process
    img = Image.open(filename)
    img = img.getdata()
    data = np.array(img,dtype='f')
    data = np.reshape(data,(28,28))
    img = [[data]]
    img = torch.tensor(img)
    img = img.type(torch.FloatTensor)

    input.append(img)
with open('processed_data.txt','wb') as fp:
	pickle.dump(input,fp)

