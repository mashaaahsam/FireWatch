# FireWatch: Fire Prediction AI Model

## Project Overview

FireWatch AI utilizes an advanced machine learning model to predict future wildfires based on 
key features identified by the same AI.

## Features

**Fire Prediction**: Predict the occurrence of wildfires based on historical data and weather conditions.

**Real-Time Heat Maps**: Visualize predicted fire locations using interactive heat maps.

## Installation

To run this project, you need to install the following libraries:
- pandas
- numpy
- joblib
- folium

```bash
pip install xgboost pandas scikit-learn folium geopandas joblib
```

## Usage

1. **Load the Dataset**: Update the file_path variable with the path to your dataset.
2. **Train the Model**: The script will preprocess the data, train the XGBoost classifier, and evaluate its performance.
3. **Generate Predictions**: The script will generate predictions for the entire dataset and create a heat map of predicted fire locations.
4. **Visualize**: The heat map will be saved as an HTML file (fire_prediction_heatmap.html).

## Code Structure

- Data Preprocessing: Handles missing values and converts feature columns to integers.
- Model Training: Splits the data into training, validation, and test sets. Trains the XGBoost model and evaluates its performance.
- Model Evaluation: Outputs accuracy, classification report, and confusion matrix.
- Model Saving: Saves the trained model using joblib.
- Visualization: Creates a heat map of predicted fire locations using Folium.
