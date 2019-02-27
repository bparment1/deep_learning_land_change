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
#DATE MODIFIED: 02/27/2019
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

def open_image(url):
    image_data = open_http_query(url)
    
    if not image_data:
            return None
            
    mmap_name = "/vsimem/"+uuid4().get_hex()
    gdal.FileFromMemBuffer(mmap_name, image_data.read())
    gdal_dataset = gdal.Open(mmap_name)
    image = gdal_dataset.GetRasterBand(1).ReadAsArray()
    gdal_dataset = None
    gdal.Unlink(mmap_name)
    
    return image

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
out_suffix = "deep_learning_houston_LUCC_02272019" #output suffix for the files and ouptut folder
#ARGS 8
NA_value = -9999 # number of cores
file_format = ".tif"

#NLCD coordinate reference system: we will use this projection rather than TX.
CRS_reg = "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
method_proj_val = "bilinear" # method option for the reprojection and resampling
gdal_installed = True #if TRUE, GDAL is used to generate distance files
prop = 0.3 # proportion used for training
random_seed = 100

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

data_fname = 'r_variables_harris_county_exercise4_02072019.txt'

#data_df = pd.read_table(os.path.join(in_dir,data_fname))
data_df = pd.read_csv(os.path.join(in_dir,data_fname))
data_df.columns

###########################################
### PART 2: Split test and train, rescaling #######


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from numpy import array

selected_covariates_names = ['land_cover', 'slope', 'roads_dist', 'developped_dist']
selected_target_names = ['change'] #also called dependent variable

selected_categorical_var_names=['land_cover']
selected_continuous_var_names=list(set(selected_covariates_names) - set(selected_categorical_var_names))

##Find frequency of unique values:
freq_val_df = data_df[selected_categorical_var_names].apply(pd.value_counts)

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
names_cat = 'lc_'.join(str(unique_val))
names_cat = 'lc_'.join(str(unique_val))
names_cat = ['lc3','lc4','lc5','lc7','lc8','lc9']

print(names_cat)
onehot_encoded_df = pd.DataFrame(onehot_encoded,columns=names_cat)
onehot_encoded_df = pd.DataFrame(onehot_encoded)
onehot_encoded_df.columns
onehot_encoded_df.head()
#onehot_encoded_df.columns = names_cat
onehot_encoded_df.shape
data_df.shape
## Combine back!!

#
## Split training and testing
#selected_covariates_names = ['land_cover', 'slope', 'roads_dist', 'developped_dist']
#selected_target_names = ['change'] #also called dependent variable
#from sklearn.cross_validation import train_test_split

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(data_df[selected_covariates_names], 
                                                    data_df[selected_target_names], 
                                                    test_size=prop, 
                                                    random_state=random_seed)

X_train.shape

#### Scaling


from sklearn.preprocessing import MinMaxScaler

# Load training data set from CSV file
#training_data_df = pd.read_csv("sales_data_training.csv")

# Load testing data set from CSV file
#test_data_df = pd.read_csv("sales_data_test.csv")

# Data needs to be scaled to a small range like 0 to 1 for the neural
# network to work well.
scaler = MinMaxScaler(feature_range=(0, 1))

##### need to use one hot encoding or text embedding to normalize categorical variables
#https://dzone.com/articles/artificial-intelligence-a-radical-anti-humanism
# Scale both the training inputs and outputs
#scaled_training = scaler.fit_transform(training_data_df)
#scaled_testing = scaler.transform(test_data_df)

scaled_training = scaler.fit_transform(X_train)
scaled_testing = scaler.transform(X_test)

type(scaled_training)
scaled_training.shape

X = scaled_training

# Print out the adjustment that the scaler applied to the total_earnings column of data
#print("Note: total_earnings values were scaled by multiplying by {:.10f} and adding {:.6f}".format(scaler.scale_[8], scaler.min_[8]))

# Create new pandas DataFrame objects from the scaled data
#scaled_training_df = pd.DataFrame(scaled_training, columns=training_data_df.columns.values)
#scaled_testing_df = pd.DataFrame(scaled_testing, columns=test_data_df.columns.values)
scaled_training_df = pd.DataFrame(scaled_training, columns=selected_covariates_names)
scaled_testing_df = pd.DataFrame(scaled_testing, columns=selected_target_names)

# Save scaled data dataframes to new CSV files
#scaled_training_df.to_csv("sales_data_training_scaled.csv", index=False)
#scaled_testing_df.to_csv("sales_data_testing_scaled.csv", index=False)

###########################################
### PART 3: build model and train #######

from keras.models import Sequential
from keras.layers import *

#training_data_df = pd.read_csv("sales_data_training_scaled.csv")

#X = training_data_df.drop('total_earnings', axis=1).values
#Y = training_data_df[['total_earnings']].values

X = X_train # to be replaced by the scaled values
Y = y_train
# Define the model

#NOTE INPUT SHOULD BE THE NUMBER OF VAR
model = Sequential()
model.add(Dense(50, input_dim=4, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', 
              optimizer='adam')

# Train the model: takes about 10 min

history = model.fit(
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

history.history['val_mean_absolute_error']
model.history.epoch # epoch
model.history.history.loss


###########################################
### PART 4: Accuracy and prediction on new data #######

# See page 81 Deep Learning book
# Load the separate test data set
#test_data_df = pd.read_csv("sales_data_test_scaled.csv")

#X_test = test_data_df.drop('total_earnings', axis=1).values
#Y_test = test_data_df[['total_earnings']].values

test_error_rate = model.evaluate(X_test, 
                                 y_test, 
                                 verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))

test_error_rate = model.evaluate(X_test, 
                                 y_test, 
                                 verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))


# Load the data we make to use to make a prediction
X = pd.read_csv("proposed_new_product.csv").values

# Make a prediction with the neural network
prediction = model.predict(X)

# Grab just the first element of the first prediction (since that's the only have one)
prediction = prediction[0][0]

# Re-scale the data from the 0-to-1 range back to dollars
# These constants are from when the data was originally scaled down to the 0-to-1 range
prediction = prediction + 0.1159
prediction = prediction / 0.0000036968

print("Earnings Prediction for Proposed Product - ${}".format(prediction))




##########################   END OF SCRIPT   ##################################















