"""
Implement the testing procedure here. 

Inputs:
    Unzip the hw4_test.zip and place the folder named "hw4_test" in the same directory of your "prediction.py" file, your "prediction.py" need to give the following required output.

Outputs:
    A file named "prediction.txt":
        * The prediction file must have 10000 lines because the testing dataset has 10000 testing images.
        * Each line is an integer prediction label (0 - 9) for the corresponding testing image.
        * The prediction results must follow the same order of the names of testing images (0.png – 9999.png).
    Notes: 
        1. The teaching staff will run your "prediction.py" to obtain your "prediction.txt" after the competition ends.
        2. The output "prediction.txt" must be the same as the final version you submitted to the CodaLab, 
        otherwise you will be given 0 score for your hw4.


**!!!!!!!!!!Important Notes!!!!!!!!!!**
    To open the folder "hw4_test" or load other related files, 
    please use open('./necessary.file') instead of open('some/randomly/local/directory/necessary.file').

    For instance, in the student Jupyter's local computer, he stores the source code like:
    - /Jupyter/Desktop/cs165B/hw4/prediction.py
    - /Jupyter/Desktop/cs165B/hw4/hw4_test
    If he/she use os.chdir('/Jupyter/Desktop/cs165B/hw4/hw4_test'), this will cause an IO error 
    when the teaching staff run his code under other system environments.
    Instead, he should use os.chdir('./hw4_test').


    If you use your local directory, your code will report an IO error when the teaching staff run your code,
    which will cause 0 score for your hw4.
"""
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import imageio
from skimage.color import rgb2gray
from skimage.transform import resize
from PIL import Image
import torchvision.models as models
import pickle
from model import *


def load_checkpoint(model, filename):
    checkpoint = torch.load(filename)
    start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.cuda()
import sys
model_path  = "./cnn2_"+str(sys.argv[1])+".pkl"
input = []
print(model_path)
with open ('processed_data.txt', 'rb') as fp:
    input = pickle.load(fp)

print(input[0][0][0][0])
model = CNN2()
model.cuda()
model.load_state_dict(torch.load(model_path))
# load_checkpoint(model,"save_model_167")
model.eval() 
output = []
for images in input:
    images = Variable(images).cuda()
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    output.append(predicted[0].item())
print(output)
print(len(output))
f = open("prediction.txt","a")
for x in output:
    f.write(str(x)+"\n")
f.close()
