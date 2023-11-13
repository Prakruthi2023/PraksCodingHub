In this introductory competition (a contest designed to help participants get started in a specific field), the task is to categorize 104 different types of flowers based on images sourced from five distinct public datasets. Some categories are quite specific, representing only a particular subtype of flower (such as pink primroses), while others encompass a broader range of subtypes (like various wild roses).
The dataset contain anomalies, such as images of flowers in unconventional settings or serving as a backdrop to modern machinery. However, overcoming these imperfections is an integral part of the challenge. The goal was to develop a classifier capable of discerning the true essence of the flowers within the images, despite any distracting elements.

Platform used to train the model:
Google Colab allows anybody to write and execute arbitrary python code through the browser, and is especially well suited to machine learning, data analysis and education. It offers free GPU usage to train the high resolution and more training dataset.

Steps involved during the training:
Importing libraries - Have imported all necessary libraries for training and visualization.
	import matplotlib.pyplot as plt
	import tensorflow as tf	
	import glob
	import numpy as np
	import math, re, os
	from tensorflow.keras import layers, models
     2. Loading dataset from Google drive which is in the format of TFRecords
	For loading and reading the TFRecord files, I have used the source code from:
       CNN Petals to the Metal | Kaggle

     3. Pre-processing the dataset -
       Resizing the images, data augmentation to increase the dataset for better accuracy, by setting 
       appropriate thresholds and linewidths, we have controlled the amount of information 
       displayed and avoiding shape mismatch error.

     4.The Model - I have used simple CNN (Convolution Neural Network) to build for the training 
     dataset. The model has 3 Conv2D layers, 2 MaxPooling2D layers and 2 Dense layers.
     Amongst the dataset folders, I tried training for192x192, 224x224 and 512x512 folder 
     dataset. 
	

Tried to plot the model loss and model accuracy using the following snippet:


Observations and final predictions:
1. With Batch Size = 16, without the learning rate optimizer, our model was underfitted with 92% accuracy and 0.2419 loss.

3. With Batch Size = 32, with the learning rate optimizer and initial rate = 0.001, our model had an improvement with 49% accuracy and 1.99 loss, though the accuracy was less, the underfitting was resolved.

4. With Batch Size = 16, with the learning rate optimizer and initial rate = 0.006, our model had an good improvement with 60% accuracy and 4.19 loss. The accuracy has increased without over or under fitting the training data.
   


My Contribution:
Implementing the model with trial-and-error methodology to conclude the number of Conv2D and MaxPooling2D layers.
Implemented the Visualzation of the accuracy of the code by referring official documentation to build the code.
Tried modifying the data loading for various image size to get the best predictions.
Modified the hyper-parameters like BATCH SIZE, EPOCH, Learning Rate to get rid of under fitting and over fitting issues.
To Navigate to my Github Source Code:

Citations:
1.CNN Petals to the Metal | Kaggle
2.https://www.tensorflow.org/tutorials/images/cnn 
3.https://www.tensorflow.org/tutorials/images/classification 
4.https://datascience.stackexchange.com/questions/64538/performances-evaluation-of-image-classification-with-different-distribution-for 
5.https://blog.infuseai.io/pops-machine-learning-workshop-1-image-classification-480276291846 
6.https://www.youtube.com/watch?v=chQNuV9B-Rw&t=740s 









