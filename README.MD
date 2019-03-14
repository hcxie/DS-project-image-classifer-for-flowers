# Table of Contents
<!-- MDTOC maxdepth:6 firsth1:1 numbering:0 flatten:0 bullets:1 updateOnSave:1 -->

- [Table of Contents](#table-of-contents)   
- [Installation](#installation)   
- [Project Motivation](#project-motivation)   
- [File descriptions](#file-descriptions)   
- [Results](#results)   
- [Licensing, Authors, Acknowledgements](#licensing-authors-acknowledgements)   

<!-- /MDTOC -->

# Installation

The required python package includes:torch,torchvision and matplotlib.
The dataset can be downloaded here:http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html

# Project Motivation
In this project is deep learning model based on VGG networks is trained to recognize different species of flowers. The deep learning package used in this project is pyTorch.
After built and trained the deep neural network, the code is converted into an application. This application is a pair of python scripts that can be run from the command line.

# File descriptions
Image Classifier Project.ipynb -> This file contains the code and results for this Project

train.py -> This is the python application that can be used to train a neural network from the command line.

predict.py -> This is the python application that can apply the trained neural network to predict the label of flower iamge.

cat_to_name.json -> This file contains the mapping between categorical labels and flower names
# Results
The test accuracy is 85%

# Licensing, Authors, Acknowledgements
Author Haochen X (jbdx6307@gmail.com)
