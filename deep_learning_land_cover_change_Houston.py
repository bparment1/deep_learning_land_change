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
#DATE MODIFIED: 04/02/2019
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
from numpy import array
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
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import *

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
out_suffix = "deep_learning_houston_LUCC_04012019" #output suffix for the files and ouptut folder
#ARGS 8
NA_value = -9999 # number of cores
file_format = ".tif"

#NLCD coordinate reference system: we will use this projection rather than TX.
CRS_reg = "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
method_proj_val = "bilinear" # method option for the reprojection and resampling
gdal_installed = True #if TRUE, GDAL is used to generate distance files
prop = 0.3 # proportion used for training
random_seed = 10 #sampling random seed

## Relevant variables used:
selected_covariates_names = ['land_cover', 'slope', 'roads_dist', 'developped_dist']
selected_target_names = ['change'] #also called dependent variable

selected_categorical_var_names=['land_cover']

### Input data files
#rastername_county_harris = "harris_county_mask.tif" #Region of interest: extent of Harris County
#elevation_fname = "srtm_Houston_area_90m.tif" #SRTM elevation
#roads_fname = "r_roads_Harris.tif" #Road count for Harris county

# -12 layers from land cover concensus (Jetz lab)
fileglob = "*.tif"
pathglob = os.path.join(in_dir, fileglob)
l_f = glob.glob(pathglob)
l_f.sort() #order input by decade
l_dir = map(lambda x: os.path.splitext(x)[0],l_f) #remmove extension
l_dir = map(lambda x: os.path.join(out_dir,os.path.basename(x)),l_dir) #set the directory output
 
	
### Aggreagate NLCD input files
infile_land_cover_date1 = "agg_3_r_nlcd2001_Houston.tif"
data_fname = 'r_variables_harris_county_exercise4_02072019.txt'
	
################# START SCRIPT ###############################

######### PART 0: Set up the output dir ################

#set up the working directory
#Create output directory

if create_out_dir==True:
    #out_path<-"/data/project/layers/commons/data_workflow/output_data"
    out_dir_new = "output_data_"+out_suffix
    out_dir = os.path.join(out_dir,out_dir_new)
    create_dir_and_check_existence(out_dir)
    os.chdir(out_dir)        #set working directory
else:
    os.chdir(create_out_dir) #use working dir defined earlier
    
    
###########################################
### PART I: READ AND VISUALIZE DATA #######
	
infile_land_cover_date1 = os.path.join(in_dir,infile_land_cover_date1) #NLCD 2001

data_df = pd.read_csv(os.path.join(in_dir,data_fname))
data_df.columns

###########################################
### PART 2: Split test and train, rescaling #######

selected_continuous_var_names=list(set(selected_covariates_names) - set(selected_categorical_var_names))
##Find frequency of unique values:
freq_val_df = data_df[selected_categorical_var_names].apply(pd.value_counts)
print(freq_val_df)

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

names_cat = ['lc_' + str(i) for i in unique_val]

print(names_cat)
onehot_encoded_df = pd.DataFrame(onehot_encoded,columns=names_cat)
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

X_train, X_test, y_train, y_test = train_test_split(data_df[selected_covariates_names_updated], 
                                                    data_df[selected_target_names], 
                                                    test_size=prop, 
                                                    random_state=random_seed)

X_train.shape

#### Scaling between 0-1 for continuous variables

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

X_testing_df.to_csv(os.path.join(out_dir,
                    "X_testing_df_"+out_suffix+".csv"))

X_training_df.to_csv(os.path.join(out_dir,
                    "X_training_df_"+out_suffix+".csv"))

###########################################
### PART 3: build model and train #######

#https://blogs.rstudio.com/tensorflow/posts/2017-12-07-text-classification-with-keras/
# binary classif see p.72

X = X_training_df.values
Y = y_train #.values

## Dealing with imbalence classes
#import keras

class_weight = {0: 0.1,
                1: 0.9}
#model.fit(X_train, Y_train, 9epochs=10, 
#          batch_size=32, class_weight=class_weight)

# Define the model

### Prepare down sampling:
train_dat = pd.DataFrame(np.concatenate(
                         (X_training_df.values,y_train.values),axis=1),
                         columns=list(X_training_df)+['change'])

train_dat_1s = train_dat[train_dat['change'] == 1]
train_dat_0s = train_dat[train_dat['change'] == 0]

### Proportion of change
prop_observed=train_dat_1s.shape[0]/train_dat_0s.shape[0]
print(prop_observed)

###
keep_0s = train_dat_0s.sample(frac=train_dat_1s.shape[0]/train_dat_0s.shape[0])
keep_0s = train_dat_0s.sample(frac=prop_observed,
                              random_state=random_seed)

#https://datascience.stackexchange.com/questions/44883/deep-network-not-able-to-learn-imbalanced-data-beyond-the-dominant-class
train_dat = pd.concat([keep_0s,train_dat_1s],axis=0)
train_dat.columns
sum(train_dat.change)/train_dat.shape[0] #50% change and no change
train_dat.shape #downsampled data

#NOTE INPUT SHOULD BE THE NUMBER OF VAR
#### Test with less number of input nodes: pruning
model1 = Sequential()
model1.add(Dense(50, input_dim=9, activation='relu'))
model1.add(Dense(100, activation='relu'))
model1.add(Dense(50, activation='relu'))
model1.add(Dense(1, activation='sigmoid'))

### model 1 wihtout validation
#model1b = keras.models.clone_model(model1)

model1b = Sequential()
model1b.add(Dense(50, input_dim=9, activation='relu'))
model1b.add(Dense(100, activation='relu'))
model1b.add(Dense(50, activation='relu'))
model1b.add(Dense(1, activation='sigmoid'))

### model 1b with validation set
#model1c = keras.models.clone_model(model1)

model1c = Sequential()
model1c.add(Dense(50, input_dim=9, activation='relu'))
model1c.add(Dense(100, activation='relu'))
model1c.add(Dense(50, activation='relu'))
model1c.add(Dense(1, activation='sigmoid'))

### model 1c with validation set and weight
model1d = keras.models.clone_model(model1)

model1d = Sequential()
model1d.add(Dense(50, input_dim=9, activation='relu'))
model1d.add(Dense(100, activation='relu'))
model1d.add(Dense(50, activation='relu'))
model1d.add(Dense(1, activation='sigmoid'))


model1.compile(loss='binary_crossentropy', #crossentropy can be optimized and is proxy for ROC AUC
              optimizer='rmsprop',
             metrics=['accuracy'])

model1b.compile(loss='binary_crossentropy', #crossentropy can be optimized and is proxy for ROC AUC
              optimizer='rmsprop',
             metrics=['accuracy'])

model1c.compile(loss='binary_crossentropy', #crossentropy can be optimized and is proxy for ROC AUC
              optimizer='rmsprop',
             metrics=['accuracy'])

model1d.compile(loss='binary_crossentropy', #crossentropy can be optimized and is proxy for ROC AUC
              optimizer='rmsprop',
             metrics=['accuracy'])

#In general the lower the crossentropy, the higher the AUC

#crossentropy measures the distance between probability distributions or in this case between 
#ground truth distribution  and the predictions

########### MODEL FITTING ###########

X = X_training_df.values
Y = y_train #.values

              
###### Use validation
              
x_validation = X[:1000] #for validation
y_validation = Y[:1000]

x_partial_train = X[1000:]
y_partial_train = Y[1000:]

X_down = train_dat.drop(columns=['change'])
Y_down = train_dat['change']

#history1_no_weight = model1.fit(
#    X,
#    Y,
#    epochs=50,
#    shuffle=True,
#    verbose=2,
#    class_weight=class_weight
#)

history1_undersampling = model1.fit(
    X_down,
    Y_down,
    epochs=50,
    #shuffle=True,
    verbose=2,
#    class_weight=class_weight
)

loss_acc_fit_model1_df = pd.DataFrame(history1_undersampling.history)
loss_acc_fit_model1_df.to_csv("loss_acc_fit_model1_undersampling_df.csv")

history1b_validation = model1b.fit(
    x_partial_train,
    y_partial_train,
    epochs=50,
    #batch=1000,
    validation_data=(x_validation,y_validation),
    verbose=2
#    class_weight=class_weight
)

loss_acc_fit_model1b_df = pd.DataFrame(history1b_validation.history)

loss_acc_fit_model1b_df.to_csv("loss_acc_fit_model1b_validation_df.csv")


###### Use validation and class weight

history1c_validation_weight = model1c.fit(
    x_partial_train,
    y_partial_train,
    epochs=50,
    #batch=1000,
    validation_data=(x_validation,y_validation),
    verbose=2,
   class_weight=class_weight
)

loss_acc_fit_model1c_df = pd.DataFrame(history1c_validation_weight.history)
loss_acc_fit_model1c_df.to_csv("loss_acc_fit_model1c_validation_weight_df.csv")

########### Down sampling

#### model 1 d undersampling
X_down = train_dat.drop(columns=['change'])
Y_down = train_dat['change']

x_validation = X_down[:1000] #for validation
y_validation = Y_down[:1000]

x_partial_train = X_down[1000:]
y_partial_train = Y_down[1000:]

history1d_undersampling = model1d.fit(
    x_partial_train,
    y_partial_train,
    epochs=50,
    #batch=1000,
    validation_data=(x_validation,y_validation),
    verbose=2,
   class_weight=class_weight
)

loss_acc_fit_model1d_df = pd.DataFrame(history1d_undersampling.history)
loss_acc_fit_model1d_df.to_csv("loss_acc_fit_model1c_validation_weight_undersampling_df.csv")

##################################
### logistic model
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
clf = LogisticRegression(random_state=0, solver='lbfgs',
                          multi_class='multinomial').fit(X, y)
model_logistic = LogisticRegression()

model_logistic = model_logistic.fit(X_train.values,y_train.values.ravel())

model_logistic.coef_
selected_covariates_names_updated

pred_test = model_logistic.predict(X_test.values)
pred_test_prob = model_logistic.predict_proba(X_test.values)

pred_test_prob[:,1] # this is the prob for 1
y_test[0:5]
pred_test_prob[0:5,:]

predicted_classes = model.predict(X)
accuracy = accuracy_score(y.flatten(),pred_test)
parameters = model.coef_
#pred_test = model_logistic.predict(X_test)

model_logistic.score(pred_test,y_test)
pred_test = model_logistic.predict_proba(X_test.values)

from sklearn.metrics import roc_auc_score

y_true = y_test
y_scores = pred_test_prob[:,1]
roc_auc_score(y_true,y_scores)

### Note that we only have about 10% change in the dataset so setting 50% does not make sense!!
sum(data_df.change)/data_df.shape[0]
sum(y_train.change)/y_train.shape[0]

#https://towardsdatascience.com/building-a-logistic-regression-in-python-301d27367c24
#This is for ROC curve
#https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8

###########################################
### PART 4: Accuracy and prediction on new data #######

#https://stackoverflow.com/questions/50115762/output-probability-score-with-keras-using-model-predict

######################
#Predictions, getting final activiation layer

np.set_printoptions(suppress=True) #prevent numpy exponential 
                                   #notation on print, default False

pred_test_model1 = pd.DataFrame(model1.predict(X_test.values),
                                columns=['pred_undersampling'])
pred_train_model1 = pd.DataFrame(model1.predict(X_train.values),
                                 columns=['pred_undersampling'])

#pred_test_model1.rename(columns={"0":"pred_undersampling"},inplace=True)

pred_test_model1.columns
pred_train_model1.max()
pred_test_model1.max()
pred_train_model1.hist()

pred_train_model1.to_csv("pred_train_all_model1_df.csv")
pred_test_model1.to_csv("pred_test_all_model1_df.csv")

#####
X_down = train_dat.drop(columns=['change'])
Y_down = train_dat['change']

X_test_down = train_dat.drop(columns=['change'])
Y_down = train_dat['change']

pred_test_down_model1 = pd.DataFrame(model1.predict(X_test.values))
pred_train_down_model1 = pd.DataFrame(model1.predict(X_down.values))

pred_train_down_model1.hist()
pred_test_down_model1.hist()

pred_train_down_model1.to_csv("pred_train_down_model1_df.csv")
pred_test_model1.to_csv("pred_test_down_model1_df.csv")

###

pred_test_model1b = pd.DataFrame(model1b.predict(X_test.values))
pred_train_model1b = pd.DataFrame(model1b.predict(X_train.values))

pred_train_model1b.max()
pred_test_model1b.max()

pred_train_model1b.to_csv("pred_train_model1b_validation_df.csv")
pred_test_model1b.to_csv("pred_test_model1b_validation_df.csv")

pred_test_model1c = pd.DataFrame(model1c.predict(X_test.values))
pred_train_model1c = pd.DataFrame(model1c.predict(X_train.values))

pred_train_model1c.max()
pred_test_model1c.max()

pred_train_model1c.to_csv("pred_train_model1c_validation_and_weight_df.csv")
pred_test_model1c.to_csv("pred_test_model1c_validation_and_weight_df.csv")

pred_test_model1d = pd.DataFrame(model1d.predict(X_test.values))
pred_train_model1d = pd.DataFrame(model1d.predict(X_train.values))

pred_train_model1d.max()
pred_test_model1d.max()

pred_train_model1d.to_csv("pred_train_all_model1d_undersampling_validation_and_weight_df.csv")
pred_test_model1d.to_csv("pred_test_all_model1d_under_sampling_validation_and_weight_df.csv")

####

X_test_down = train_dat.drop(columns=['change'])
Y_down = train_dat['change']

pred_test_down_model1d = pd.DataFrame(model1d.predict(X_test.values),
                                      columns=["pred_test"])
pred_train_down_model1d = pd.DataFrame(model1.predict(X_down.values),
                                       columns=["pred_train_down"])

pred_train_down_model1d['change'] = Y_down.values
pred_train_down_model1d.columns

pred_test_down_model1d['change'] = y_test.values

pred_test_down_model1d['pred_test'].hist()
pred_train_down_model1d['change'].hist()

pred_test_down_model1d.change.hist()


pred_train_down_model1d.to_csv("pred_train_down_model1_undersampling_validation_and_weight_df2.csv")
pred_test_down_model1d.to_csv("pred_test_all_model1_undersampling_validation_and_weight_df.csv")

pred_test_model1d = pd.DataFrame(model1d.predict(X_test.values))
pred_train_model1d = pd.DataFrame(model1d.predict(X_train.values))

evaluate(X_test, 
         y_test, 
         verbose=0)
         
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))


#https://www.dlology.com/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/

poch_step = np.arange(1,51,1)

type(history1.history) # this is a dictionary
history1.history['acc']
history1.history['loss']
history1.epoch

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

# See page 81 Deep Learning book

#test_error_rate = model


##########################   END OF SCRIPT   ##################################















