"""
Script Function: This script loads interpolated elevation data, calculates various features, trains a
Random Forest model, and predicts future elevation changes. It handles data gaps and ensures
robust feature calculation.

Required Libraries:
- pandas
- numpy
- os
- logging
- sklearn
- tqdm

To install the required libraries, you can use the following pip commands:
pip install pandas numpy scikit-learn tqdm
"""

import pandas as pd
import numpy as np
import os
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Setup for logging progress in real-time
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_interpolated_data(directory, years, quarters=None):
    """
        Loads interpolated data for specified years and quarters from the given directory.

        :param directory: Directory containing the interpolated data files.
        :param years: List of years to load data for.
        :param quarters: List of quarters to load data for. If None, loads data for the entire year.
        :return: Combined DataFrame containing all loaded data.
        """
    data_frames = []
    for year in tqdm(years, desc="Loading data for years"):
        if quarters:
            for quarter in tqdm(quarters, desc=f"Loading data for {year}", leave=False):
                file_path = os.path.join(directory, f'interpolated_{year}_Q{quarter}.xyz')
                if os.path.exists(file_path):
                    logging.info(f"Loading data from {file_path}")
                    df = pd.read_csv(file_path, delim_whitespace=True, header=None,
                                     names=['Easting', 'Northing', 'Elevation'])
                    df['Year'] = year
                    df['Quarter'] = quarter
                    data_frames.append(df)
                else:
                    logging.warning(f"No data found for {year} Q{quarter}. Skipping.")
        else:
            file_path = os.path.join(directory, f'interpolated_{year}.xyz')
            if os.path.exists(file_path):
                logging.info(f"Loading data from {file_path}")
                df = pd.read_csv(file_path, delim_whitespace=True, header=None,
                                 names=['Easting', 'Northing', 'Elevation'])
                df['Year'] = year
                df['Quarter'] = 1  # Default to Q1 if quarters are not specified
                data_frames.append(df)
            else:
                logging.warning(f"No data found for {year}. Skipping.")
    if data_frames:
        return pd.concat(data_frames, ignore_index=True)
    else:
        logging.error("No data frames to concatenate. Returning empty DataFrame.")
        return pd.DataFrame()


# Function to calculate the required spatio-temporal features
def calculate_features(data):
    """
        Calculates various features for the given data.

        :param data: DataFrame containing the interpolated elevation data.
        :return: DataFrame with calculated features.
        """
    if data.empty:
        logging.warning("No data to calculate features.")
        return data

    logging.info("Calculating features...")

    # Ensure the data is sorted by spatial coordinates (Easting, Northing) and temporal coordinates (Year, Quarter).
    # This sorting is crucial for calculating features that depend on the order of time (e.g., differences over time).
    data.sort_values(by=['Easting', 'Northing', 'Year', 'Quarter'], inplace=True)

    # Convert the 'Elevation' column to a numeric type to avoid non numeric entries
    data['Elevation'] = pd.to_numeric(data['Elevation'], errors='coerce')

    # Calculate the rate of elevation change:
    # This is done by grouping the data by spatial coordinates (Easting, Northing),
    # then taking the difference in elevation between consecutive time points (diff()),
    # and dividing it by the time difference (Year diff in quarters + Quarter diff).
    data['Rate_of_Elevation_Change'] = data.groupby(['Easting', 'Northing'])['Elevation'].diff() / (
        (data.groupby(['Easting', 'Northing'])['Year'].diff() * 4 + data.groupby(['Easting',
                                                                                  'Northing'])['Quarter'].diff()))

    # Calculate cumulative elevation change:
    # This sums the elevation changes over time for each spatial point, showing the elevation change cumulatively.
    data['Cumulative_Elevation_Change'] = data.groupby(['Easting', 'Northing'])['Elevation'].cumsum()

    # Calculate the elevation lag (previous quarter's elevation):
    # Shifts the elevation column by one time step within each spatial group, providing the previous quarter's elevation
    data['Elevation_Lag1'] = data.groupby(['Easting', 'Northing'])['Elevation'].shift(1)

    # Calculate the difference between the current elevation and the elevation from the same quarter in the previous
    # year. This helps in identifying cyclical or seasonal changes in elevation.
    data['Sandwave_Change'] = data['Elevation'] - data.groupby(['Easting', 'Northing'])['Elevation'].shift(4)

    # Topographic and Statistical Calculations

    # Local slope: Estimates the slope (rate of change) of elevation over the surrounding area for each point.
    data['Local_Slope'] = calculate_local_slope(data)

    # Local curvature: Measures the curvature (rate of change of the slope) of the elevation at each point.
    data['Local_Curvature'] = calculate_local_curvature(data)

    # Omni-directional gradient: Calculates the gradient of the elevation in all directions, providing a measure of
    # steepness.
    data['Omni_Directional_Gradient'] = calculate_omni_directional_gradient(data)

    # Elevation difference with neighbor: Calculates the elevation difference between a point and its nearest neighbors.
    data['Elevation_Diff_Neighbor'] = calculate_elevation_diff_neighbor(data)

    # Aspect ratio: Computes the aspect ratio, which is a measure of the elongation or flatness of the surface.
    data['Aspect_Ratio'] = calculate_aspect_ratio(data)

    # Roughness: Measures the roughness of the elevation change by calculating the localized standard deviation
    # of elevation at each point in 2D space.
    data['Roughness'] = calculate_roughness(data)

    # Rolling mean: Computes a rolling mean of elevation over the last four quarters, providing a smoothed elev. value.
    data['Rolling_Mean'] = data.groupby(['Easting', 'Northing'])['Elevation'].transform(lambda x: x.rolling(
        window=4, min_periods=1).mean())

    # Rolling standard deviation: Computes the rolling standard deviation of elevation over the last four quarters,
    # which gives an estimate of the variability of elevation changes.
    data['Rolling_Std'] = data.groupby(['Easting', 'Northing'])['Elevation'].transform(lambda x: x.rolling(
        window=4, min_periods=1).std())

    # Topographic Position Index (TPI): Measures the relative position of a point compared to its surrounding terrain.
    data['TPI'] = calculate_tpi(data)

    # Vertical exaggeration: Provides an exaggerated measure of elevation differences to quantify the terrain steepness.
    data['Vertical_Exaggeration'] = calculate_vertical_exaggeration(data)

    # Interaction features: These features combine spatial coordinates with temporal information.
    data['Easting_Quarter_Interaction'] = data['Easting'] * data['Quarter']
    data['Northing_Quarter_Interaction'] = data['Northing'] * data['Quarter']

    # Data error handling
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(0, inplace=True)
    data.clip(lower=-1e6, upper=1e6, inplace=True)

    logging.info("Feature calculation completed.")
    return data


def calculate_local_slope(data):
    slopes = np.zeros(len(data))

    # Group the data by 'Easting' and 'Northing' to calculate slope for each spatial location
    for (easting, northing), group in tqdm(data.groupby(['Easting', 'Northing']), desc="Calculating local slope",
                                           leave=False):
        # Only calculate the slope if there are multiple time points for the location (at least 2)
        if len(group) > 1:
            # Sort the group by time (Year and Quarter) to ensure proper temporal order
            group_sorted = group.sort_values(by=['Year', 'Quarter'])

            # Calculate the gradient (slope) of the elevation over time for the location
            slopes[group_sorted.index] = np.gradient(group_sorted['Elevation'])
    return slopes


def calculate_local_curvature(data):
    curvatures = np.zeros(len(data))
    for (easting, northing), group in tqdm(data.groupby(['Easting', 'Northing']),
                                           desc="Calculating local curvature", leave=False):

        # Only calculate curvature if there are enough time points for the location (at least 3)
        if len(group) > 2:
            # Sort the group by time (Year and Quarter) to ensure proper temporal order
            group_sorted = group.sort_values(by=['Year', 'Quarter'])

            # Calculate the first derivative (slope) of the elevation
            first_derivative = np.gradient(group_sorted['Elevation'])

            # Calculate the second derivative (curvature) of the elevation
            curvatures[group_sorted.index] = np.gradient(first_derivative)
    return curvatures


def calculate_omni_directional_gradient(data):
    gradients = np.zeros(len(data))
    for (easting, northing), group in tqdm(data.groupby(['Easting', 'Northing']),
                                           desc="Calculating omni-directional gradient", leave=False):

        # Only calculate the gradient if there are multiple time points for the location (at least 2)
        if len(group) > 1:
            group_sorted = group.sort_values(by=['Year', 'Quarter'])

            # Calculate differences in easting, northing, and elevation
            easting_diff = group_sorted['Easting'].diff().fillna(1)
            northing_diff = group_sorted['Northing'].diff().fillna(1)
            elevation_diff = group_sorted['Elevation'].diff().fillna(0)

            # Calculate the omni-directional gradient (magnitude of slope considering all directions)
            gradients[group_sorted.index] = np.sqrt((elevation_diff / easting_diff)**2 +
                                                    (elevation_diff / northing_diff)**2)
    return gradients


def calculate_elevation_diff_neighbor(data):
    diffs = np.zeros(len(data))
    for (easting, northing), group in tqdm(data.groupby(['Easting', 'Northing']),
                                           desc="Calculating elevation difference with neighbor", leave=False):
        if len(group) > 1:
            group_sorted = group.sort_values(by=['Year', 'Quarter'])

            # Calculate the difference in elevation from the previous time point
            diffs[group_sorted.index] = group_sorted['Elevation'] - group_sorted['Elevation'].shift(1).fillna(0)

    return diffs


def calculate_aspect_ratio(data):
    ratios = np.zeros(len(data))
    for (easting, northing), group in tqdm(data.groupby(['Easting', 'Northing']), desc="Calculating aspect ratio",
                                           leave=False):
        if len(group) > 1:
            group_sorted = group.sort_values(by=['Year', 'Quarter'])
            max_elevation = group_sorted['Elevation'].max()
            min_elevation = group_sorted['Elevation'].min()

            # Aspect ratio is the ratio of maximum to minimum elevation (1e-6 added to avoid dividing by zero)
            ratios[group_sorted.index] = max_elevation / (min_elevation + 1e-6)
    return ratios


def calculate_roughness(data):
    roughness = np.zeros(len(data))
    for (easting, northing), group in tqdm(data.groupby(['Easting', 'Northing']), desc="Calculating roughness",
                                           leave=False):

        # Iterate over each group of points with the same Easting and Northing (i.e., the same spatial location) for
        # standard deviation calculation over time
        if len(group) > 1:
            group_sorted = group.sort_values(by=['Year', 'Quarter'])
            roughness[group_sorted.index] = np.std(group_sorted['Elevation'])
    return roughness


def calculate_tpi(data):
    tpi = np.zeros(len(data))
    for (easting, northing), group in tqdm(data.groupby(['Easting', 'Northing']), desc="Calculating TPI", leave=False):
        if len(group) > 1:
            group_sorted = group.sort_values(by=['Year', 'Quarter'])
            mean_elevation = group_sorted['Elevation'].mean()

            # TPI is the difference between the current elevation and the mean elevation
            tpi[group_sorted.index] = group_sorted['Elevation'] - mean_elevation
        return tpi


def calculate_vertical_exaggeration(data):
    exaggeration = np.zeros(len(data))
    for (easting, northing), group in tqdm(data.groupby(['Easting', 'Northing']),
                                           desc="Calculating vertical exaggeration", leave=False):

        # Iterate over each group of points with the same Easting and Northing (i.e., the same spatial location)
        if len(group) > 1:
            group_sorted = group.sort_values(by=['Year', 'Quarter'])
            max_elevation = group_sorted['Elevation'].max()
            min_elevation = group_sorted['Elevation'].min()

            # Calculate the horizontal distance spanned by the group (difference between max and min Easting values)
            horizontal_distance = group_sorted['Easting'].max() - group_sorted['Easting'].min()

            # Calculate vertical exaggeration as the ratio of the elevation difference to the horizontal distance.
            # Adding a small constant (1e-6) to the horizontal distance prevents division by zero.
            exaggeration[group_sorted.index] = (max_elevation - min_elevation) / (horizontal_distance + 1e-6)

    return exaggeration


def prepare_data_for_model(data):
    features = [
        'Rate_of_Elevation_Change',
        'Cumulative_Elevation_Change',
        'Elevation_Lag1',
        'Quarter',
        'Sandwave_Change',
        'Local_Slope',
        'Local_Curvature',
        'Omni_Directional_Gradient',
        'Elevation_Diff_Neighbor',
        'Aspect_Ratio',
        'Roughness',
        'Rolling_Mean',
        'Rolling_Std',
        'TPI',
        'Vertical_Exaggeration',
        'Easting',
        'Northing',
        'Easting_Quarter_Interaction',
        'Northing_Quarter_Interaction'
    ]
    X = data[features]
    y = data['Elevation']
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_random_forest(X_train, y_train):
    """
        Trains a Random Forest Regressor model using the training data.

        :param X_train: Training features
        :param y_train: Training target variable
        :return: Trained Random Forest model
        """
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
        Evaluates the Random Forest model and logs the performance metrics.

        :param model: Trained Random Forest model
        :param X_test: Testing features
        :param y_test: Testing target variable
        :return: predictions, mae, rmse, r2, std_dev, outliers, outlier_percentage
        """
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    std_dev = np.std(y_test - predictions)

    upper_threshold = predictions + 2 * std_dev
    lower_threshold = predictions - 2 * std_dev
    outliers = np.sum((y_test > upper_threshold) | (y_test < lower_threshold))
    total = len(y_test)
    outlier_percentage = (outliers / total) * 100

    logging.info(f'MAE: {mae}, RMSE: {rmse}, R-squared: {r2}, Std Dev of Errors: {std_dev}')
    logging.info(f'Number of outliers: {outliers}, Percentage of outliers: {outlier_percentage:.2f}%')

    return predictions, mae, rmse, r2, std_dev, outliers, outlier_percentage


def save_predictions_as_xyz(predictions, data, year, output_directory):
    """
        Saves the model predictions as a .xyz file.

        :param predictions: Predicted elevation values
        :param data: DataFrame containing the original data
        :param year: Year for which the predictions were made
        :param output_directory: Directory to save the predictions file
        """
    result_df = pd.DataFrame({
        'Easting': data['Easting'],
        'Northing': data['Northing'],
        'Predicted_Elevation': predictions
    })
    output_file = os.path.join(output_directory, f'predicted_elevations_{year}.xyz')
    result_df.to_csv(output_file, sep=' ', index=False, header=False)
    logging.info(f"Saved predictions for {year} to {output_file}")


def generate_future_features(current_data, predicted_data, year, quarter):
    """
        Generates future features based on the current data and the model's predictions.

        :param current_data: DataFrame containing the current data
        :param predicted_data: Predicted elevation values
        :param year: Year for the future prediction
        :param quarter: Quarter for the future prediction
        :return: DataFrame with future features
        """
    future_data = current_data.copy()
    future_data['Year'] = year
    future_data['Quarter'] = quarter
    future_data['Elevation'] = predicted_data
    future_data['Elevation_Lag1'] = future_data['Elevation']
    future_data['Rate_of_Elevation_Change'] = future_data.groupby(['Easting', 'Northing'])['Elevation'].diff() / (
        (future_data.groupby(['Easting', 'Northing'])['Year'].diff() * 4
         + future_data.groupby(['Easting', 'Northing'])['Quarter'].diff()))
    future_data['Sandwave_Change'] = (future_data['Elevation']
                                      - future_data.groupby(['Easting', 'Northing'])['Elevation'].shift(4))
    future_data['Local_Slope'] = calculate_local_slope(future_data)
    future_data['Local_Curvature'] = calculate_local_curvature(future_data)
    future_data['Omni_Directional_Gradient'] = calculate_omni_directional_gradient(future_data)
    future_data['Elevation_Diff_Neighbor'] = calculate_elevation_diff_neighbor(future_data)
    future_data['Aspect_Ratio'] = calculate_aspect_ratio(future_data)
    future_data['Roughness'] = calculate_roughness(future_data)
    future_data['Rolling_Mean'] = future_data.groupby(['Easting', 'Northing'])['Elevation'].transform(
        lambda x: x.rolling(window=4, min_periods=1).mean())
    future_data['Rolling_Std'] = future_data.groupby(['Easting', 'Northing'])['Elevation'].transform(
        lambda x: x.rolling(window=4, min_periods=1).std())
    future_data['TPI'] = calculate_tpi(future_data)
    future_data['Vertical_Exaggeration'] = calculate_vertical_exaggeration(future_data)
    future_data['Easting_Quarter_Interaction'] = future_data['Easting'] * future_data['Quarter']
    future_data['Northing_Quarter_Interaction'] = future_data['Northing'] * future_data['Quarter']
    future_data.fillna(0, inplace=True)
    return future_data


if __name__ == '__main__':
    train_years = [2016, 2017, 2018, 2019]    # Enter training years
    # test_years = [2020]                     # Enter test years (Optional)
    future_years = [2020, 2021, 2022, 2023]   # Enter Future years for Prediction
    quarters = [2, 3, 4]         # Enter available Quarter epochs with at least 3 available instances to compute metrics

    logging.info("Loading training data...")
    train_data = load_interpolated_data('Processed_Data', train_years, quarters=quarters)
    # Change surface grid files directory as required

    if train_data.empty:
        logging.error("No training data found. Exiting.")
    else:
        logging.info("Calculating features for training data...")
        train_data = calculate_features(train_data)

        logging.info("Preparing training and testing datasets...")
        X_train, X_test, y_train, y_test = prepare_data_for_model(train_data)

        logging.info("Training Random Forest model...")
        rf_model = train_random_forest(X_train, y_train)

        evaluate_model(rf_model, X_train, y_train)
        evaluate_model(rf_model, X_test, y_test)

        output_directory = 'Predicted_Data'  # Change Output Directory as required for Predicted grids
        os.makedirs(output_directory, exist_ok=True)

        # Uncomment if using testing datasets
        """for year in test_years:
            for quarter in quarters:
                logging.info(f"Loading testing data for {year} Q{quarter}...")
                test_data = load_interpolated_data('Processed_Data', [year], quarters=[quarter])

                if test_data.empty:
                    logging.warning(f"No testing data found for {year} Q{quarter}. Skipping.")
                    continue

                logging.info(f"Calculating features for testing data for {year} Q{quarter}...")
                test_data = calculate_features(test_data)

                X_test = test_data[
                    ['Rate_of_Elevation_Change', 'Cumulative_Elevation_Change', 'Elevation_Lag1', 'Quarter',
                     'Sandwave_Change', 'Local_Slope', 'Local_Curvature', 'Omni_Directional_Gradient',
                     'Elevation_Diff_Neighbor', 'Aspect_Ratio', 'Roughness', 'Rolling_Mean', 'Rolling_Std', 'TPI',
                     'Vertical_Exaggeration', 'Easting', 'Northing', 'Easting_Quarter_Interaction',
                     'Northing_Quarter_Interaction']]
                y_test = test_data['Elevation']

                logging.info(f"Evaluating model for {year} Q{quarter}...")
                predictions, mae, rmse, r2, std_dev, outliers, outlier_percentage = evaluate_model(
                    rf_model, X_test, y_test)

                logging.info(f"Saving predictions for {year} Q{quarter}...")
                save_predictions_as_xyz(predictions, test_data, f"{year}_Q{quarter}", output_directory)

                test_data['Elevation'] = predictions
                train_data = pd.concat([train_data, test_data], ignore_index=True)
                train_data = calculate_features(train_data)"""

        # Appending previous epoch into the prediction process
        last_train_year = train_data['Year'].max()
        last_train_quarter = train_data['Quarter'].max()
        future_data = train_data[(train_data['Year'] == last_train_year) & (
                train_data['Quarter'] == last_train_quarter)].copy()

        for future_year in future_years:
            for quarter in quarters:
                future_year_fraction = future_year + (quarter - 1) / 4
                logging.info(f"Generating features for future year {future_year_fraction} Q{quarter}...")
                if future_data.empty:
                    logging.warning(f"No data to generate features for {future_year_fraction}. Skipping.")
                    continue

                X_future = future_data[
                    ['Rate_of_Elevation_Change', 'Cumulative_Elevation_Change', 'Elevation_Lag1', 'Quarter',
                     'Sandwave_Change', 'Local_Slope', 'Local_Curvature', 'Omni_Directional_Gradient',
                     'Elevation_Diff_Neighbor', 'Aspect_Ratio', 'Roughness', 'Rolling_Mean', 'Rolling_Std', 'TPI',
                     'Vertical_Exaggeration', 'Easting', 'Northing', 'Easting_Quarter_Interaction',
                     'Northing_Quarter_Interaction']]

                y_future = future_data['Elevation']

                logging.info(f"Predicting elevations for {future_year_fraction}...")
                future_predictions = rf_model.predict(X_future)

                # Bias adjustment to maintain consistency
                bias_adjustment = np.mean(y_train) - np.mean(future_predictions)
                future_predictions += bias_adjustment

                save_predictions_as_xyz(future_predictions, future_data, f"{future_year}_Q{quarter}",
                                        output_directory)

                future_data = generate_future_features(future_data, future_predictions, future_year, quarter)

                # Update train_data with the new future_data for the next prediction
                train_data = pd.concat([train_data, future_data], ignore_index=True)
                train_data = calculate_features(train_data)

            logging.info("Prediction and feature generation completed.")
