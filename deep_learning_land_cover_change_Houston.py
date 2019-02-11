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
#DATE MODIFIED: 02/11/2019
#Version: 1
#PROJECT: AAG 2019
#TO DO:
#
#COMMIT: initial commit
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
out_suffix = "deep_learning_houston_LUCC_02072019" #output suffix for the files and ouptut folder
#ARGS 8
NA_value = -9999 # number of cores
file_format = ".tif"

#NLCD coordinate reference system: we will use this projection rather than TX.
CRS_reg = "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
method_proj_val = "bilinear" # method option for the reprojection and resampling
gdal_installed = True #if TRUE, GDAL is used to generate distance files
		
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
infile_land_cover_date2 = "agg_3_r_nlcd2006_Houston.tif"
infile_land_cover_date3 = "agg_3_r_nlcd2011_Houston.tif"
	
infile_name_nlcd_legend = "nlcd_legend.txt"
infile_name_nlcd_classification_system = "classification_system_nlcd_legend.xlsx"
	
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
    
    
#######################################
### PART 1: Read in DATA #######

###########################################
### PART I: READ AND VISUALIZE DATA #######
	
infile_land_cover_date1 = os.path.join(in_dir,infile_land_cover_date1) #NLCD 2001
infile_land_cover_date2 = os.path.join(in_dir,infile_land_cover_date2) #NLCD 2006
infile_land_cover_date3 = os.path.join(in_dir,infile_land_cover_date3) #NLCD 2011

lc_date1 = rasterio.open(infile_land_cover_date1) 
r_lc_date1 = lc_date1.read(1,masked=True) #read first array with masked value, nan are assigned for NA
lc_date2 = rasterio.open(infile_land_cover_date2) 
r_lc_date2 = lc_date2.read(1,masked=True) #read first array with masked value, nan are assigned for NA
lc_date3= rasterio.open(infile_land_cover_date2) 
r_lc_date3 = lc_date3.read(1,masked=True) #read first array with masked value, nan are assigned for NA

spatial_extent = rasterio.plot.plotting_extent(lc_date1)
plot.show(r_lc_date1)

#Note that you can also plot the raster io data reader
type(lc_date2)
plot.show(lc_date2)

lc_date1.crs # not defined with *.rst
lc_legend_df = pd.read_table(os.path.join(in_dir,infile_name_nlcd_legend),sep=",")
	
lc_legend_df.head() # Inspect data
plot.show(lc_date2) # View NLCD 2006, we will need to add the legend use the appropriate palette!!
plot.show(lc_date2,cmap=plt.cm.get_cmap('cubehelix',16 ))	
### Let's generate a palette from the NLCD legend information to view the existing land cover for 2006.
#names(lc_legend_df)
lc_legend_df.columns
lc_legend_df.shape
#dim(lc_legend_df) #contains a lot of empty rows
	
#lc_legend_df<- subset(lc_legend_df,COUNT>0) #subset the data to remove unsured rows
lc_legend_df = lc_legend_df[lc_legend_df['COUNT']>0] #subset the data to remove unsured rows


# Generate palette

colors_val = ['linen', 'lightgreen', 'darkgreen', 'maroon']

cmap = colors.ListedColormap(colors_val) # can be used directly
webcolors.rgb_to_name
webcolors.rgb_to_name((0, 0, 0)) #default is css3 convention
webcolors.rgb_to_name((255,255,255))
webcolors.name_to_rgb('navy')

### Generate a palette color from the input Red, Green and Blue information using RGB encoding:
rgb_col=list(zip(lc_legend_df.Red,lc_legend_df.Green,lc_legend_df.Blue))
len(rgb_col)
rgb_col[0]
#lc_legend_df$rgb <- paste(lc_legend_df$Red,lc_legend_df$Green,lc_legend_df$Blue,sep=",") #combine
#','.join([lc_legend_df.Red,lc_legend_df.Green, lc_legend_df.Blue]) 	

#lc_legend_df['rgb'] = lc_legend_df[['Red','Green','Blue']].apply[lambda x:]
lc_legend_df['rgb'] = rgb_col
### row 2 correspond to the "open water" category
webcolors.rgb_to_name(rgb_col[1])

color_val_water = rgb_col[1]
color_val_developed_high = rgb_col[7]


#######
################################################
###  PART II : Analyze change and transitions

## As the plot shows for 2006, we have 15 land cover types. Analyzing such complex categories in terms of decreasse (loss), increase (gain), 
# persistence in land cover will generate a large number of transitions (potential up to 15*15=225 transitions in this case!)

## To generalize the information, let's aggregate leveraging the hierachical nature of NLCD Anderson Classification system.


data_fname = 'r_variables_harris_county_exercise4_02072019.txt'

#data_df = pd.read_table(os.path.join(in_dir,data_fname))
data_df = pd.read_csv(os.path.join(in_dir,data_fname))
data_df.columns




###################### END OF SCRIPT #####################

















