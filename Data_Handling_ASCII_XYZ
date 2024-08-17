"""
This script processes non-uniform XYZ files to interpolate and filter elevation data within a specified border.
The interpolation method used is Nearest Neighbor.

Necessary Libraries:
- pandas: For handling data frames
- numpy: For numerical operations
- os: For file and directory operations
- re: For regular expressions to parse filenames
- scipy: For interpolation methods
- shapely: For geometric operations

To install the required libraries, you can use the following pip commands:
pip install pandas numpy scipy shapely
"""
import pandas as pd
import numpy as np
import os
import re
from scipy.interpolate import NearestNDInterpolator
from shapely.geometry import Point, Polygon


# Function to load XYZ data from a file
def load_xyz(file_path):
    """
        Loads XYZ data from a given file path.

        Args:
            file_path (str): Path to the XYZ file.

        Returns:
            pd.DataFrame: DataFrame containing the loaded data with columns ['Easting', 'Northing', 'Elevation'].
        """
    print(f"Loading data from {file_path}...")
    try:
        data = pd.read_csv(file_path, sep='\s+', header=None, names=['Easting', 'Northing', 'Elevation'],
                           dtype={'Easting': np.float32, 'Northing': np.float32, 'Elevation': np.float32})
        return data
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None


# Function to extract the year and quarter from a filename
def extract_date_from_filename(filename):
    """
        Extracts the year and quarter from a filename using regular expressions.

        Args:
            filename (str): Filename to extract information from.

        Returns:
            tuple: (year, quarter) if successful, otherwise (None, None).
        """
    match = re.search(r'ALBERT_(\d{4})_Q(\d)_Filtered', filename)
    # Change file name as required replacing "(\d{4})_Q(\d)" with "YYYY_QX" where X is the required Quarter (i.e. Q2)
    if match:
        year = int(match.group(1))
        quarter = int(match.group(2))
        return year, quarter
    print(f"Filename {filename} does not match expected format.")
    return None, None


# Function to find the common boundary of all input datasets
def find_common_boundary(input_directory):
    """
        Determines the common boundary (min/max Easting and Northing) across all XYZ files in the input directory.

        Args:
            input_directory (str): Directory containing the XYZ files.

        Returns:
            tuple: (min_easting, max_easting, min_northing, max_northing)
        """
    print("Determining the common boundary for all datasets...")
    min_easting, max_easting = float('inf'), float('-inf')
    min_northing, max_northing = float('inf'), float('-inf')

    # Iterate over all files in the directory to determine the boundary limits
    for filename in os.listdir(input_directory):
        if filename.endswith('.xyz'):
            data = load_xyz(os.path.join(input_directory, filename))
            if data is not None:
                min_easting = min(min_easting, data['Easting'].min())
                max_easting = max(max_easting, data['Easting'].max())
                min_northing = min(min_northing, data['Northing'].min())
                max_northing = max(max_northing, data['Northing'].max())
    print(f"Common boundary set to Easting: {min_easting} to {max_easting}, Northing: {min_northing} to {max_northing}")
    return min_easting, max_easting, min_northing, max_northing


# Function to interpolate data using Nearest Neighbor interpolation
def interpolate_data(data, grid_x, grid_y):
    print("Interpolating data using NearestNDInterpolator...")
    try:
        points = data[['Easting', 'Northing']].values  # Coordinates of source points
        values = data['Elevation'].values              # Elevations of source points
        interpolator = NearestNDInterpolator(points, values)
        grid_z = interpolator(grid_x, grid_y)
        return grid_z
    except Exception as e:
        print(f"Interpolation failed: {e}")
        return None


# Function to load the border file and create a polygon
def load_border_file(border_file_path):
    """
       Loads an ASCII format border file and creates a polygon for filtering.

       Args:
           border_file_path (str): Path to the border file.

       Returns:
           Polygon: Shapely Polygon object representing the border.
       """
    print(f"Loading border file from {border_file_path}...")
    try:
        border_df = pd.read_csv(border_file_path, sep='\s+', header=None, names=['Easting', 'Northing'])
        border_polygon = Polygon(border_df.values)
        return border_polygon
    except Exception as e:
        print(f"Failed to load border file {border_file_path}: {e}")
        return None


# Function to filter data by border polygon
def filter_data_by_border(grid_x, grid_y, grid_z, border_polygon):
    """
        Filters the interpolated elevation data to include only points within a specified polygon border.

        Args:
            grid_x (np.ndarray): 1D array of X coordinates (Easting) for the grid.
            grid_y (np.ndarray): 1D array of Y coordinates (Northing) for the grid.
            grid_z (np.ndarray): 1D array of interpolated elevation values.
            border_polygon (Polygon): Shapely Polygon object representing the border.

        Returns:
            tuple: Filtered arrays of (grid_x, grid_y, grid_z) within the polygon.
        """
    print("Filtering interpolated data based on border...")
    points = np.array([Point(easting, northing) for easting, northing in zip(grid_x.ravel(), grid_y.ravel())])
    mask = np.array([border_polygon.contains(point) for point in points])
    return grid_x.ravel()[mask], grid_y.ravel()[mask], grid_z.ravel()[mask]


# Function to smooth data by averaging neighboring points
def smooth_data(grid_x, grid_y, grid_z, threshold=0.5):
    """
       Smooths the interpolated data by averaging neighboring points to remove spikes.

       Args:
           grid_x (np.ndarray): 2D array of X coordinates (Easting) for the grid.
           grid_y (np.ndarray): 2D array of Y coordinates (Northing) for the grid.
           grid_z (np.ndarray): 2D array of interpolated elevation values.
           threshold (float): Difference threshold to detect spikes.

       Returns:
           np.ndarray: Smoothed 2D array of elevation values.
       """
    print("Smoothing data to remove spikes...")
    for i in range(1, grid_z.shape[0] - 1):
        for j in range(1, grid_z.shape[1] - 1):
            neighbors = grid_z[i - 1:i + 2, j - 1:j + 2].ravel()
            center_value = grid_z[i, j]

            # If any neighboring point has a large difference from the center, average the points
            if np.any(np.abs(neighbors - center_value) > threshold):
                grid_z[i, j] = np.mean(neighbors)
    return grid_z


# Main function to process files sequentially
def process_files_sequentially(input_directory, output_directory, border_file_path, grid_resolution=0.5):
    """
       Processes all XYZ files in the input directory by interpolating and filtering based on a specified border.

       Args:
           input_directory (str): Directory containing raw XYZ files.
           output_directory (str): Directory to save the processed files.
           border_file_path (str): Path to the border file.
           grid_resolution (float): Resolution of the grid for interpolation. # Change according to data availability

       Returns:
           None
       """
    # Load the border polygon
    border_polygon = load_border_file(border_file_path)
    if border_polygon is None:
        print("No valid border file. Exiting.")
        return

    # Determine the common boundary for the datasets
    min_easting, max_easting, min_northing, max_northing = find_common_boundary(input_directory)

    # Create a mesh grid based on the boundary and grid resolution
    grid_x, grid_y = np.meshgrid(
        np.arange(min_easting, max_easting, grid_resolution),
        np.arange(min_northing, max_northing, grid_resolution),
        indexing='ij'
    )
    print("Starting file processing...")

    # Iterate over each XYZ file in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.xyz'):
            file_path = os.path.join(input_directory, filename)
            data = load_xyz(file_path)
            if data is not None:
                grid_z = interpolate_data(data, grid_x, grid_y)
                if grid_z is not None:
                    grid_z = smooth_data(grid_x, grid_y, grid_z)
                    grid_x_trimmed, grid_y_trimmed, grid_z_trimmed = filter_data_by_border(grid_x, grid_y, grid_z,
                                                                                           border_polygon)
                    year, quarter = extract_date_from_filename(filename)
                    if year and quarter:
                        output_file_path = os.path.join(output_directory, f"interpolated_{year}_Q{quarter}.xyz")
                        np.savetxt(output_file_path, np.column_stack([grid_x_trimmed, grid_y_trimmed, grid_z_trimmed]),
                                   fmt='%f')
                        print(f"Saved trimmed interpolated data to {output_file_path}")


if __name__ == '__main__':
    input_directory = 'Raw_xyz'  # Change Directory of folder for loading the raw XYZ files as required
    output_directory = 'Processed_Data'  # Change Directory of folder for saving processed XYZ files as required
    border_file_path = 'border.brd'  # Change border file path and filename as required
    print("Starting processing of datasets...")
    process_files_sequentially(input_directory, output_directory, border_file_path)
    print("Processing complete.")
