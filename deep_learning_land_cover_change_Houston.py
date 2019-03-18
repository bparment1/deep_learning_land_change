# -*- coding: utf-8 -*-
"""
Spyder Editor.
"""
#################################### Land Use and Land Cover Change #######################################
############################ Analyze Land Cover change in Houston #######################################
#This script generate a deep learning model using variables generated to track and predict land change in Houston.
#The goal is to assess land cover change using two/three land cover maps in the Houston areas.
#Additional datasets are provided for the land cover change modeling. A model is built for Harris county.
#
#AUTHORS: Benoit Parmentier
#DATE CREATED: 02/07/2019
#DATE MODIFIED: 03/18/2019
#Version: 1
#PROJECT: AAG 2019
#TO DO:
#
#COMMIT: adding input parameters and some clean up
#
#################################################################################################
	
	
###### Library used in this script

import gdal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import seaborn as sns
import rasterio
import subprocess
import pandas as pd
import os, glob
from rasterio import plot
import geopandas as gpd
import descartes
import pysal as ps
from cartopy import crs as ccrs
from pyproj import Proj
from osgeo import osr
from shapely.geometry import Point
from collections import OrderedDict
import webcolors
import sklearn
import keras

################ NOW FUNCTIONS  ###################

##------------------
# Functions used in the script 
##------------------

def create_dir_and_check_existence(path):
    #Create a new directory
    try:
        os.makedirs(path)
    except:
        print ("directory already exists")

############################################################################
#####  Parameters and argument set up ########### 

#ARGS 1
#in_dir = "/home/bparmentier/c_drive/Users/bparmentier/Data/python/Exercise_4/data"
in_dir = "/home/bparmentier/c_drive/Users/bparmentier/Data/AAG/deeplearning/land_change_Houston_deep_learning/data"
#ARGS 2
#out_dir = "/home/bparmentier/c_drive/Users/bparmentier/Data/python/Exercise_4/outputs"
out_dir = "/home/bparmentier/c_drive/Users/bparmentier/Data/AAG/deeplearning/land_change_Houston_deep_learning/outputs"

#ARGS 3:
create_out_dir=True #create a new ouput dir if TRUE
#ARGS 7
out_suffix = "deep_learning_houston_LUCC_03182019" #output suffix for the files and ouptut folder
#ARGS 8
NA_value = -9999 # number of cores
file_format = ".tif"

#NLCD coordinate reference system: we will use this projection rather than TX.
CRS_reg = "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
method_proj_val = "bilinear" # method option for the reprojection and resampling
gdal_installed = True #if TRUE, GDAL is used to generate distance files
prop = 0.3 # proportion used for training
random_seed = 100 #sampling random seed

## Relevant variables used:
selected_covariates_names = ['land_cover', 'slope', 'roads_dist', 'developped_dist']
selected_target_names = ['change'] #also called dependent variable

selected_categorical_var_names=['land_cover']

### Input data files
#rastername_county_harris = "harris_county_mask.tif" #Region of interest: extent of Harris County
#elevation_fname = "srtm_Houston_area_90m.tif" #SRTM elevation
#roads_fname = "r_roads_Harris.tif" #Road count for Harris county

# Raster variables 
fileglob = "*.tif"
pathglob = os.path.join(in_dir, fileglob)
l_f = glob.glob(pathglob)
l_f.sort() #order input by decade
l_dir = map(lambda x: os.path.splitext(x)[0],l_f) #remmove extension
l_dir = map(lambda x: os.path.join(out_dir,os.path.basename(x)),l_dir) #set the directory output
 
	
### Aggreagate NLCD input files
infile_land_cover_date1 = "agg_3_r_nlcd2001_Houston.tif"
### dataset as csv
data_fname = 'r_variables_harris_county_exercise4_02072019.txt'
	
################# START SCRIPT ###############################

######### PART 0: Set up the output dir ################

#set up the working directory
#Create output directory

if create_out_dir==True:
    #out_path<-"/data/project/layers/commons/data_workflow/output_data"
    out_dir = "output_data_"+out_suffix
    out_dir = os.path.join(in_dir,out_dir)
    create_dir_and_check_existence(out_dir)
    os.chdir(out_dir)        #set working directory
else:
    os.chdir(create_out_dir) #use working dir defined earlier
    
    
###########################################
### PART I: READ AND VISUALIZE DATA #######
	
infile_land_cover_date1 = os.path.join(in_dir,infile_land_cover_date1) #NLCD 2001

#data_df = pd.read_table(os.path.join(in_dir,data_fname))
data_df = pd.read_csv(os.path.join(in_dir,data_fname))
data_df.columns

###########################################
### PART 2: Split test and train, rescaling and normalization #######

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from numpy import array

selected_continuous_var_names=list(set(selected_covariates_names) - set(selected_categorical_var_names))
##Find frequency of unique values:
freq_val_df = data_df[selected_categorical_var_names].apply(pd.value_counts)
print(freq_val_df.head())

values_cat = array(data_df[selected_categorical_var_names].values) #note this is assuming only one cat val here

label_encoder = LabelEncoder() 
one_hot_encoder = OneHotEncoder(sparse=False)

### First integer encode:
integer_encoded = label_encoder.fit_transform(values_cat)
print(integer_encoded)

# Binary encode:
integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
print(integer_encoded)

onehot_encoded = one_hot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
onehot_encoded.shape
type(onehot_encoded)

#invert to check value?
onehot_encoded[0:5,]
values_cat[0:5,]

inverted = label_encoder.inverse_transform([np.argmax(onehot_encoded[0,:])])
inverted = label_encoder.inverse_transform([np.argmax(onehot_encoded[1,:])])
print(inverted)

#assign back to the data.frame

unique_val = np.array(freq_val_df.index)
unique_val = np.sort(unique_val)

print(unique_val)

#names_cat = map(lambda x: 'lc_'+str(x),unique_val) #remmove extension
names_cat = ['lc_' + str(i) for i in unique_val]

print(names_cat)
onehot_encoded_df = pd.DataFrame(onehot_encoded,
                                 columns=names_cat)
onehot_encoded_df.columns
onehot_encoded_df.head()
onehot_encoded_df.shape
data_df.shape
## Combine back!!

data_df= pd.concat([data_df,onehot_encoded_df],sort=False,axis=1)
data_df.shape
data_df.head()

selected_covariates_names_updated = selected_continuous_var_names + names_cat 

## Split training and testing

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_df[selected_covariates_names_updated], 
                                                    data_df[selected_target_names], 
                                                    test_size=prop, 
                                                    random_state=random_seed)

X_train.shape

#### Scaling between 0-1 for continuous variables

from sklearn.preprocessing import MinMaxScaler

# Data needs to be scaled to a small range like 0 to 1 for the neural
# network to work well.
scaler = MinMaxScaler(feature_range=(0, 1))
### need to select only the continuous var:
scaled_training = scaler.fit_transform(X_train[selected_continuous_var_names])
scaled_testing = scaler.transform(X_test[selected_continuous_var_names])

type(scaled_training) # array
scaled_training.shape

## Concatenate column-wise
X_testing_df = pd.DataFrame(np.concatenate((X_test[names_cat].values,scaled_testing),axis=1),
                                            columns=names_cat+selected_continuous_var_names)
X_training_df = pd.DataFrame(np.concatenate((X_train[names_cat].values,scaled_training),axis=1),
                                            columns=names_cat+selected_continuous_var_names)

###########################################
### PART 3: build model and train #######

from keras.models import Sequential
from keras.layers import *

#https://blogs.rstudio.com/tensorflow/posts/2017-12-07-text-classification-with-keras/
# binary classif see p.72

X = X_training_df.values
Y = y_train #.values

#import keras

class_weight = {0: 0.25,
                1: 0.75}
#model.fit(X_train, Y_train, epochs=10, 
#          batch_size=32, class_weight=class_weight)


# Define the model

### TRy down sampling:
train_dat = y_train

train_dat_1s = train_dat[train_dat['change'] == 1]

train_dat_0s = train_dat[train_dat['change'] == 0]
keep_0s = train_dat_0s.sample(frac=train_dat_1s.shape[0]/train_dat_0s.shape[0])

train_dat = pd.concat([keep_0s,train_dat_1s],axis=0)

#NOTE INPUT SHOULD BE THE NUMBER OF VAR
#### Test with less number of input nodes: pruning
model1 = Sequential()
model1.add(Dense(50, input_dim=9, activation='relu'))
model1.add(Dense(100, activation='relu'))
model1.add(Dense(50, activation='relu'))
model1.add(Dense(1, activation='sigmoid'))

model1.compile(loss='binary_crossentropy', #crossentropy can be optimized and is proxy for ROC AUC
              optimizer='rmsprop',
             metrics=['accuracy'])

#### Test with less number of input nodes: pruning
model2 = Sequential()
model2.add(Dense(25, input_dim=9, activation='relu'))
model2.add(Dense(50, activation='relu')) 
model2.add(Dense(25, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))
#model.compile(loss='binary_crossentropy', 
#              optimizer='adam',
#              metrics=['accuracy'])

model2.compile(loss='binary_crossentropy', #crossentropy can be optimized and is proxy for ROC AUC
              optimizer='rmsprop',
              metrics=['accuracy'])

#In general the lower the crossentropy, the higher the AUC

#crossentropy measures the distance between probability distributions or in this case between 
#ground truth distribution  and the predictions

history1_no_weight = model1.fit(
    X,
    Y,
    epochs=50,
    shuffle=True,
    verbose=2,
#    class_weight=class_weight
)
              
history1 = model1.fit(
    X,
    Y,
    epochs=50,
    shuffle=True,
    verbose=2,
    class_weight=class_weight
)

#https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer
# Train the model: takes about 10 min
from keras import backend as K

inp = model1.input                                           # input placeholder
outputs = [layer.output for layer in model1.layers]          # all layer outputs
functors = [K.function([inp, K.learning_phase()], [out])

You can easily get the outputs of any layer by using: 
index=1
lay_4 = model1.layers[index].output

history2 = model2.fit(
    X,
    Y,
    epochs=50,
    shuffle=True,
    verbose=2
)

### Note you should add a validation dataset!!!
##See book p.89 Deep learning with python

#num_epochs = 50
#history = model.fit(
#    partial_train_data,
#    partial_train_target,
#    validation_data=(val_data,val_targets)
#    epochs=num_epochs,
#    batch_size=1,
#    verbose=0)
#)

epoch_step = np.arange(1,51,1)

type(history1.history) # this is a dictionary
history1.history['acc']
history1.history['loss']
history1.epoch

#test=pd.DataFrame(np.array((epoch_step,history2.history['acc'],history2.history['loss'])).T)
test=pd.DataFrame({'epoch':epoch_step,
                   'acc':history1.history['acc'],
                   'loss':history1.history['loss']})

test.shape
test.head()
#history.history['val_mean_absolute_error']
#model.history.epoch # epoch
#model.history.history.loss

plt.plot(test['epoch'],test['acc'])
#plt.plot(test['acc'])
 
# multiple line plot
plt.plot( 'epoch', 'acc', 
         data=test, 
         marker='o', markerfacecolor='blue', 
         markersize=12, 
         color='skyblue', linewidth=4)
plt.plot( 'epoch', 'loss', 
         data=test, 
         marker='', 
         color='olive', linewidth=2)

#plt.plot( 'x', 'y3', data=df, 
#         marker='', color='olive', 
#         linewidth=2, linestyle='dashed', label="toto")
#plt.legend()

##################################
### logistic model
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
clf = LogisticRegression(random_state=0, solver='lbfgs',
                          multi_class='multinomial').fit(X, y)
model_logistic = LogisticRegression()

model_logistic = model_logistic.fit(X_train.values,y_train.values.ravel())

#/usr/local/lib/python3.5/dist-packages/sklearn/utils/validation.py:761: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
#  y = column_or_1d(y, warn=True)
#Out[103]: 
#LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#          intercept_scaling=1, max_iter=100, multi_class='warn',
#          n_jobs=None, penalty='l2', random_state=None, solver='lbfgs',
#          tol=0.0001, verbose=0, warm_start=False)


#model_logistic.fit(X_train.values,y_train.values)
#yy=y_train.values.reshape(y_train.shape[0],1)
#model_logistic.fit(X_train.values,yy)

pred_test = model_logistic.predict(X_test.values)
pred_test_prob = model_logistic.predict_proba(X_test.values)

#Note:
sum(pred_test)

pred_test_prob[:,1] # this is the prob for 1
y_test[0:5]
pred_test_prob[0:5,:]

#pred_test = model_logistic.predict(X_test)

model_logistic.score(pred_test,y_test)
pred_test = model_logistic.predict_proba(X_test.values)

from sklearn.metrics import roc_auc_score

y_true = y_test
y_scores = pred_test_prob[:,1]
roc_auc_score(y_true,y_scores)

#https://towardsdatascience.com/building-a-logistic-regression-in-python-301d27367c24
#https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8

###########################################
### PART 4: Accuracy and prediction on new data #######

# See page 81 Deep Learning book

test_error_rate = model1.evaluate(X_test, 
                                 y_test, 
                                 verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))

#https://www.dlology.com/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/

from keras.wrappers.scikit_learn import KerasClassifier

tt1_test1=model1.predict_proba(X_test.values)
tt1_test2=model1.predict(X_test.values)


from sklearn.metrics import roc_curve
y_pred_keras = model1.predict(X_test.values).ravel()

fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)

from sklearn.metrics import roc_auc_score

roc_auc_score(y_test,y_pred_keras)

# Load the data we make to use to make a prediction
#X = pd.read_csv("proposed_new_product.csv").values

# Make a prediction with the neural network
#prediction = model.predict(X)

# Grab just the first element of the first prediction (since that's the only have one)
#prediction = prediction[0][0]

# Re-scale the data from the 0-to-1 range back to dollars
# These constants are from when the data was originally scaled down to the 0-to-1 range
#prediction = prediction + 0.1159
#prediction = prediction / 0.0000036968

#print("Earnings Prediction for Proposed Product - ${}".format(prediction))

##########################   END OF SCRIPT   ##################################















