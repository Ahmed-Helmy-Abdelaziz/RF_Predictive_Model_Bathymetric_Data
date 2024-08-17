# RF_Predictive_Model_Bathymetric_Data

**Riverbed Geomorphology Prediction and Analysis**

This repository contains Python scripts developed for the analysis and prediction of riverbed geomorphological changes, specifically focused on hydrographic survey bathymetric data. These scripts are designed to handle data processing, predictive modeling, and surface difference analysis, providing a comprehensive toolkit for analyzing riverbed dynamics over time.

This work was done in partial fullfilment of CEGE0049:Research Project module for UCL's MSc Geospatial Sciences (Hydrographic Surveying). 

**Scripts Overview**

This is a brief overview of the scripts, with full details commented in each of the scripts.

**Data_Handling_ASCII_XYZ.py**

This script is responsible for preparing and processing bathymetric data in ASCII XYZ format. It includes functions for loading, cleaning, and interpolating pre-cleaned survey data surfaces into uniform 0.5m x 0.5m resolution grids, ensuring that each epoch of data is projected on the same fixed grid for use in predictive modeling.

**RF_Predictive_Model.py**

This script implements a Random Forest predictive model to forecast future riverbed elevations. It handles the training of the model, feature extraction, and sequential prediction of future surfaces based on historical data. The script is designed to manage large datasets and produce reliable predictions with minimal overfitting.

**Surface_Difference_Analysis.py**

This script performs a detailed analysis of the differences between predicted and actual riverbed surfaces. It calculates key metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Total Volume Difference, providing insights into the model's accuracy and highlighting areas of significant deviation.

**Usage**

**Data Preparation**

Start by running the Data_Handling_ASCII_XYZ.py script to process your raw survey data. This will prepare the data for predictive modeling by cleaning and interpolating it into a consistent format.

**Predictive Modeling**

Use the RF_Predictive_Model.py script to train the Random Forest model on your prepared data. This script will generate predictions for future riverbed elevations based on historical patterns.

**Surface Difference Analysis**

After generating predictions, run the Surface_Difference_Analysis.py script to compare the predicted surfaces with actual measurements. This will help you assess the accuracy of your model and identify areas where the predictions deviate from reality.

**Python Package Requirements**

Python 3.x

pandas

numpy

scikit-learn

matplotlib

scipy

Make sure to install the required packages before running the scripts.

**Data Requirements**

1- Processed and cleaned historical XYZ surfaces for a designated survey area boundary (Survey grids should be copied to "Raw_xyz" folder in the project directory as per current setting)

2- ASCII Border file for the survey boundary (To be placed in project folder by the name "border.brd" as per current settings)

3- At least 3 epochs of survey grids are required to calculate training features for the current RF predictive model, otherwise remove Quarter number with insufficient data.

4- If the actual measured datasets for the set predicted future years XYZ grids are available, surface difference analysis script can be run to evaluate the model predictive performance visually and statistically
