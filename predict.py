#python predict.py flowers/test/1/image_06743.jpg mymodel_checkpoint.pth

#import package here
import argparse
import torch
from torchvision import transforms, models
from torch import nn, optim
import torch.nn.functional as F
import json
import numpy as np
from PIL import Image
import numpy as np


def checkpointfunction(args):
    #load the checkpoint.pth file, the input needs to be a string of checkpoint
    #name
    newcheckpoint=torch.load(args.checkpoint)
    
    if newcheckpoint['arch'] == 'vgg16':
        newmymodel = models.vgg16(pretrained=True)
        #freeze parameters in mynewmodel
        for param in newmymodel.parameters():
            param.requires_grad=False
    else:
        print("only vgg16 structure is allowed")
    
    newmymodel.class_to_idx = newcheckpoint['class_to_idx']
    
    if len(newcheckpoint['hidden_layer']) == 3:
        newmymodel.classifier=nn.Sequential(nn.Linear(newcheckpoint['input_size'],newcheckpoint['hidden_layer'][0]),
                              nn.ReLU(),
                              nn.Dropout(0.2),
                              nn.Linear(newcheckpoint['hidden_layer'][0],newcheckpoint['hidden_layer'][1]),
                              nn.ReLU(),
                              nn.Dropout(0.2),
                              nn.Linear(newcheckpoint['hidden_layer'][1],newcheckpoint['hidden_layer'][2]),
                              nn.ReLU(),
                              nn.Dropout(0.2),
                              nn.Linear(newcheckpoint['hidden_layer'][2],newcheckpoint['output_size']),
                              nn.LogSoftmax(dim=1))
        newmymodel.load_state_dict(newcheckpoint['state_dict'])
    else:
        print("the checkpoint needs to have 3 hidden layer")
    
    return newmymodel

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    Image_loader = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    pil_image = Image.open(image)
    pil_image = Image_loader(pil_image)
                               
    return pil_image

def predict_flower(args, newmymodel):
    image_inputs = process_image(args.input)
   
    image_inputs=image_inputs.unsqueeze(0)
    image_inputs = image_inputs.float()
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu == True else "cpu")
    newmymodel.to(device)
    #image_inputs.to(device)
    
    if torch.cuda.is_available() and args.gpu == True :
        with torch.no_grad():
            ps = torch.exp(newmymodel.forward(image_inputs.cuda()))
    else:
        with torch.no_grad():
            ps = torch.exp(newmymodel.forward(image_inputs))
        
    top_p, top_class = ps.topk(args.top_k, dim=1)
    
    index = 1
    import json

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    #in the cat_to_name the label index starts from 1. However, in top_class, the index starts from 0, therefore, index+1
    probability = np.array(top_p[0])
    labels = [cat_to_name[str(index+1)] for index in np.array(top_class[0])]
    #print(labels)
    #print(cat_to_name)
    #print(np.array(top_class[0]))
    
    return probability, labels
    
def main():
    parser = argparse.ArgumentParser(description='Flower Image Classifcation Model Predictor')
    #Define arguments
    parser.add_argument('input', type=str, help='image to process and predict')
    parser.add_argument('checkpoint', type=str, help='cnn to load')
    parser.add_argument('--top_k', type=int, default=3, help='default top_k results')
    parser.add_argument('--category_names',  type=str, default='cat_to_name.json',help='default category file' )
    parser.add_argument('--gpu', type=bool, default=True, help='Use GPU or not')
    args = parser.parse_args()
    
    newmymodel=checkpointfunction(args)
    probablility, labels = predict_flower(args, newmymodel)
    predictions = list(zip(labels, probablility))
    for i in range(len(predictions)):
        print('{} : {:.3%}'.format(predictions[i][0], predictions[i][1]))
     
   
        
if __name__ == '__main__':
    main()