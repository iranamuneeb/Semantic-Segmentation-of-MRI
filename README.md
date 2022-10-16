# Semantic-Segmentation-of-MRI

## Problem
The main problem with today's MRI machine is the number of errors and wrong images they can generate while processing information. <br> 
Which can lead to wrong diagnosis or false positive/ false negative diagnosis of the patient. <br>

# Solution
We used our own Generator to generate and Use images in our custom 3D-Unet model while converting our images into numpy arrays for faster & better results. 

## Approach

* Step 1: Get the data ready ( Data_processing.py)
* Step 2: Define custom data generator (genarator.py)
* Step 3: Define the 3D U-net model  (Simple_3d_unet.py)
* Step 4: Train and Predict  (train.py)

## How To Use

<b>Download the data set</b>
> [Data Set](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation/code?resource=download)

```bash
# Clone this repository
$ git clone https://github.com/iranamuneeb/Semantic-Segmentation-of-MRI

# Install the required packages
$ pip install segmentation-models-3D
$ pip install split-folders 
$ pip install tensorflow 
$ pip install nibabel #(nibabel is a must for the data_processing.py to utilize NII formated files and convert them into numpy arrays)


# Before starting the project make sure all the files and folder are in place!
```

> **Note**
> In BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_355 one image file is in labeled wrong, correct it before proceeding further. 

## Working

For first the NII files are combined and converted into an Numpy array for an easier and better use. <br>
Then we have created our own custom Genarator that utilizes the the numpy arrays to convert them into usable input for our 3D_unet mode. <br>
Then the Unet model creats a merged layer upon which the model is trained with the parameters and epochs. 


> **Note**
> For a detailed explaination please wait for our paper

> Our system took 8 hours to train one epoch with CPU utilization. And the dataset takes upto 100GB of space. 

> A web application supporting this ML model will be live soon as well. 

## Credits
<b> The Lofrzz </B>
* Rana Muneeb Asad 
* Taimoor Wajid
* Farjad Khan

> **Reach Out To us**
> You can reach out to us on muneebasad.24@gmail.com for any querry
