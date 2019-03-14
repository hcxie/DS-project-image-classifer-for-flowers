#python train.py

#import package and library
import torch
from torchvision import datasets,transforms,models
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import argparse

#some global parameter
data_batch_size = 64


#data preparetion
def load_data(args):
    #create data directory for training, validation and testing set
    data_dir = args.data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #crop, flip, rotation, normalize the data and make the image to tensor
    data_transforms = {"train":transforms.Compose([transforms.RandomRotation(30),
                                   transforms.RandomResizedCrop(224),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485,0.456,0.406],
                                                    [0.229,0.224,0.225])]),
                   "validation":transforms.Compose([transforms.RandomRotation(30),
                                   transforms.RandomResizedCrop(224),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485,0.456,0.406],
                                                    [0.229,0.224,0.225])]),
                   "test":transforms.Compose([transforms.Resize(255),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                   [0.229, 0.224, 0.225])])}
    #apply the data transform to image file    
    image_datasets = {"train" : datasets.ImageFolder(train_dir,transform=data_transforms['train']),
                  "validation": datasets.ImageFolder(valid_dir, transform=data_transforms['validation']),
                  "test":datasets.ImageFolder(test_dir, transform=data_transforms['test'])}
    #create data loaders    
    dataloaders = {"train":torch.utils.data.DataLoader(image_datasets['train'], batch_size=data_batch_size, shuffle=True),
               "validation":torch.utils.data.DataLoader(image_datasets['validation'], batch_size=data_batch_size),
               "test":torch.utils.data.DataLoader(image_datasets['test'],batch_size=data_batch_size)}
        
    return dataloaders, image_datasets
    
    #subfunction to construct a model
def model_build(args):
    
    #the model arch is based on vgg 16. it has 3 hidden layer
        if args.arch == 'vgg16':
            mymodel=models.vgg16(pretrained=True)
			input_size = mymodel.classifier[0].in_features
        elif args.arch == 'densenet121':
           mymodel=models.densenet121(pretrained=True)
		   input_size = mymodel.classifier.in_features
        else:
            print("please use either  vgg16 or densenet121")
            
        for param in mymodel.parameters():
            param.requires_grad=False
        
        hidden_units1=args.hidden_units1
        hidden_units2=args.hidden_units2
        hidden_units3=args.hidden_units3
        
		
		
		
        mymodel.classifier=nn.Sequential(nn.Linear(input_size,hidden_units1),
                              nn.ReLU(),
                              nn.Dropout(0.2),
                              nn.Linear(hidden_units1,hidden_units2),
                              nn.ReLU(),
                              nn.Dropout(0.2),
                              nn.Linear(hidden_units2,hidden_units3),
                              nn.ReLU(),
                              nn.Dropout(0.2),
                              nn.Linear(hidden_units3,102),
                              nn.LogSoftmax(dim=1))
        
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(mymodel.classifier.parameters(), lr=args.lr)
        device = torch.device("cuda" if torch.cuda.is_available() and args.gpu == True else "cpu")
        mymodel.to(device)
        
        return mymodel, device, criterion, optimizer
    
def training_model(args,dataloaders,image_datasets,mymodel,device,criterion,optimizer):
    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in dataloaders['train']:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = mymodel.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                mymodel.eval()
                with torch.no_grad():
                    for inputs, labels in dataloaders['validation']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = mymodel.forward(inputs)
                        batch_loss = criterion(logps, labels)
                    
                        test_loss += batch_loss.item()
                    
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                    print(f"Epoch {epoch+1}/{epochs}.. "
                            f"Train loss: {running_loss/print_every:.3f}.. "
                            f"Validation loss: {test_loss/len(dataloaders['validation']):.3f}.. "
                            f"Validation accuracy: {accuracy/len(dataloaders['validation']):.3f}")
            running_loss = 0
            mymodel.train()
    #create checkpoint                
    mymodel.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint={'input_size':25088,
                'output_size':102,
                'hidden_layer':[args.hidden_units1,args.hidden_units2,args.hidden_units3],
                'state_dict':mymodel.state_dict(),
                'class_to_idx':mymodel.class_to_idx,
                'opimizer':optimizer.state_dict(),
                'epochs_num':args.epochs,
                'arch':'vgg16'
           }
    torch.save(checkpoint, args.save_dir)    

def main():
    parser = argparse.ArgumentParser(description='Flower Image Classifcation Model Trainer')
    parser.add_argument('--gpu', type=bool, default=True, help='Use GPU or not')
    parser.add_argument('--arch', type=str, default='vgg16', help='please use either  vgg16 or densenet121'')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--hidden_units1', type=int, default=4096, help='hidden units for fc layer1')
    parser.add_argument('--hidden_units2', type=int, default=4096, help='hidden units for fc layer2')
    parser.add_argument('--hidden_units3', type=int, default=510, help='hidden units for fc layer3')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--data_directory', type=str, default='flowers', help='dataset directory')
    parser.add_argument('--save_dir' , type=str, default='mymodel_checkpoint.pth', help='path of your saved model')
    args = parser.parse_args()
    
    import json
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    dataloaders, image_datasets=load_data(args)
    mymodel, device, criterion, optimizer = model_build(args)
    training_model(args,dataloaders,image_datasets,mymodel,device, criterion, optimizer)
    
if __name__ == "__main__":
    main()
    
    
     