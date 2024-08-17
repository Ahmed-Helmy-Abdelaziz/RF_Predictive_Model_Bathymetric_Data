"""
XYZ Surface Difference Analysis Script

This script compares two surface elevation datasets, interpolates elevations, calculates differences,
visualizes these differences, computes statistics, and saves the results to files.

Required Libraries:
- pandas
- os
- matplotlib
- scipy
- sklearn

To install the required libraries, run:
pip install pandas matplotlib scipy scikit-learn

Usage:
1. Rename your surface elevation files to ('surface1.xyz' and 'surface2.xyz') where the resulted difference model
 will represent (surface1 - surface 2)
2. Place the 2 surface XYZ files in a folder named 'Difference_Analysis' within the project folder.
3. Run this script.

The results will be saved in the same 'Difference_Analysis' folder.
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.colors as mcolors
from sklearn.linear_model import LinearRegression


# Function to load data from a .xyz file into a DataFrame
def load_xyz(file_path):
    return pd.read_csv(file_path, sep='\s+', header=None, names=['Easting', 'Northing', 'Elevation'])


# Function to interpolate elevations of the source DataFrame to the coordinates of the target DataFrame
def interpolate_to_grid(df_source, df_target):
    points = df_source[['Easting', 'Northing']].values  # Coordinates of source points
    values = df_source['Elevation'].values  # Elevations of source points
    target_points = df_target[['Easting', 'Northing']].values  # Coordinates of target points

    # Interpolate source elevations to target coordinates
    interpolated_values = griddata(points, values, target_points, method='linear')
    df_target['Interpolated_Elevation'] = interpolated_values

    return df_target


# Function to calculate elevation differences between two DataFrames
def calculate_difference(df1, df2):
    df2 = interpolate_to_grid(df2, df1)  # Interpolate df2 elevations to df1 coordinates
    df = df1.copy()
    df['Elevation_Measured'] = df['Elevation']  # Original elevations
    df['Elevation_Predicted'] = df2['Interpolated_Elevation']  # Interpolated elevations
    df['Elevation_Difference'] = df['Elevation_Predicted'] - df['Elevation_Measured']  # Difference

    return df.dropna(subset=['Elevation_Predicted'])  # Remove rows with NaN values


# Function to create a custom diverging colormap for visualization
def create_custom_colormap():
    colors = [(0, 0, 1), (0.9, 0.9, 0.9), (1, 0, 0)]  # blue -> light gray -> red
    n_bins = 100  # Discretize the colormap into bins
    cmap_name = 'custom_diverging'
    cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    return cmap


# Function to visualize elevation differences on a scatter plot
def visualize_difference(df, output_path):
    plt.figure(figsize=(10, 8))
    cmap = create_custom_colormap()
    norm = mcolors.TwoSlopeNorm(vmin=df['Elevation_Difference'].min(), vcenter=0, vmax=df['Elevation_Difference'].max())
    plt.scatter(df['Easting'], df['Northing'], c=df['Elevation_Difference'], cmap=cmap, norm=norm, s=10, marker='o')
    plt.colorbar(label='Elevation Difference (m)')
    plt.title('Elevation Difference Between Two Surfaces')
    plt.xlabel('Easting')
    plt.ylabel('Northing')
    plt.grid(True)
    plt.savefig(output_path)
    plt.show()


# Function to compute various statistics on the elevation differences
def compute_statistics(df):
    median_diff = df['Elevation_Difference'].median()
    mean_diff = df['Elevation_Difference'].mean()
    std_diff = df['Elevation_Difference'].std()

    mae = df['Elevation_Difference'].abs().mean()  # Absolute Mean
    rmse = (df['Elevation_Difference'] ** 2).mean() ** 0.5  # Root Mean Squared Error

    # Mean Absolute Percentage Error
    mape = (df['Elevation_Difference'].abs() / df['Elevation_Measured'].abs()).mean() * 100

    tolerance_value = 1.96 * std_diff  # 95% Confidence Interval Tolerance

    total_area = df.shape[0] * (0.5 * 0.5)  # Assuming each cell represents 0.5m x 0.5m area
    total_volume = df['Elevation_Difference'].sum() * (0.5 * 0.5)

    # Outliers calculation (at 95% Confidence Level)
    lower_bound = mean_diff - 1.96 * std_diff
    upper_bound = mean_diff + 1.96 * std_diff
    outliers = df[(df['Elevation_Difference'] < lower_bound) | (df['Elevation_Difference'] > upper_bound)]
    num_outliers = outliers.shape[0]
    perc_outliers = (num_outliers / df.shape[0]) * 100

    # Linear regression to calculate R-squared
    reg = LinearRegression().fit(df[['Elevation_Measured']], df['Elevation_Predicted'])
    r_squared = reg.score(df[['Elevation_Measured']], df['Elevation_Predicted'])
    n = df.shape[0]
    p = 1  # Number of predictors

    stats = {
        'Median (m)': median_diff,
        'Mean (m)': mean_diff,
        'Standard Deviation (m)': std_diff,
        'Mean Absolute Error (MAE) (m)': mae,
        'Root Mean Squared Error (RMSE) (m)': rmse,
        'Mean Absolute Percentage Error (MAPE) (%)': mape,
        'Tolerance Value (95% confidence) (m)': tolerance_value,
        'Total 2D Surface Area (m2)': total_area,
        'Total Volume Difference (m3)': total_volume,
        'Number of Outliers': num_outliers,
        'Percentage of Outliers (%)': perc_outliers,
        'R-squared': r_squared,
    }
    return stats


# Function to plot regression graph between measured and predicted elevations
def plot_regression(df, output_path, r_squared):
    plt.figure(figsize=(10, 8))
    plt.scatter(df['Elevation_Measured'], df['Elevation_Predicted'], color='grey', s=10, alpha=0.4, label='Data points')

    # Add 95% confidence thresholds
    std_diff = df['Elevation_Difference'].std()
    mean_diff = df['Elevation_Difference'].mean()
    lower_threshold = df['Elevation_Measured'] + (mean_diff - 1.96 * std_diff)
    upper_threshold = df['Elevation_Measured'] + (mean_diff + 1.96 * std_diff)

    plt.fill_between(df['Elevation_Measured'], lower_threshold, upper_threshold, color='black', alpha=0.2,
                     label='95% Confidence Interval')

    plt.plot([df['Elevation_Measured'].min(), df['Elevation_Measured'].max()],
             [df['Elevation_Measured'].min(), df['Elevation_Measured'].max()], color='red', linewidth=2,
             label='Best fit')

    # Add R-squared value to the plot in the bottom right corner with LaTeX formatting
    plt.text(0.95, 0.05, f'$R^2 = {r_squared:.4f}$', transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='bottom', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))

    plt.xlabel('Measured Elevation')
    plt.ylabel('Predicted Elevation')
    plt.title('Measured vs Predicted Scatter Plot')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.show()


# Function to save statistics to an Excel file
def save_statistics_to_excel(stats, output_path):
    stats_df = pd.DataFrame(stats, index=[0])
    stats_df.to_excel(output_path, index=False)


# Main function to execute the workflow
def main():
    folder_path = 'Difference_Analysis'                # Change to different file path as required
    file1 = os.path.join(folder_path, 'surface1.xyz')  # Change to different file name and path for the top surface
    file2 = os.path.join(folder_path, 'surface2.xyz')  # Change to different file name and path for subtracted surface

    df1 = load_xyz(file1)  # Load first surface data
    df2 = load_xyz(file2)  # Load second surface data

    print("Calculating differences...")
    df_diff = calculate_difference(df1, df2)  # Calculate elevation differences

    print("Visualizing differences...")
    visualize_difference(df_diff, os.path.join(folder_path, 'elevation_difference.png'))  # Visualize differences

    print("Computing statistics...")
    stats = compute_statistics(df_diff)  # Compute statistics on the differences

    print("Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("Saving statistics to Excel...")
    save_statistics_to_excel(stats, os.path.join(folder_path, 'statistics.xlsx'))  # Save statistics to Excel

    print("Plotting regression graph...")
    plot_regression(df_diff, os.path.join(folder_path, 'regression_graph.png'), stats['R-squared'])  # Plot Graph


# Entry point of the script
if __name__ == '__main__':
    main()
