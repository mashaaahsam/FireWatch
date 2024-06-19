# Install necessary libraries
pip install xgboost pandas scikit-learn folium geopandas joblib


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import folium
from folium.plugins import HeatMap


# Load the dataset
file_path = "path/to/your/dataset.csv"
data = pd.read_csv(file_path)


# Preprocessing: Handling missing values
data.dropna(inplace=True)
y = data.fire_indicator.tolist()

# Key Features to train on
features = ['temperature', 'relative_humidity', 'wind_speed', 'fire_month', 'fire_hour', 'fire_date', 'fire_location_latitude', 'fire_location_longitude']

def col2int(data_f, feature):
    col_list = data_f[feature].tolist()
    col_list = [int(round(x)) for x in col_list]
    return col_list

data_features = pd.DataFrame()
for f in features:
    data_features[f] = col2int(data, f)


# Split the dataset into training (80%), validation (10%), and test sets (10%)
X_train, X_temp, y_train, y_temp = train_test_split(data_features, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize the XGBClassifier
model = XGBClassifier()

# Train the model
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=True)

# Cross-validation to check for model stability
cross_val_scores = cross_val_score(model, data_features, y, cv=5)
print("Cross-Validation Scores:", cross_val_scores)
print("Mean Cross-Validation Score:", np.mean(cross_val_scores))

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Detailed classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the trained model
joblib.dump(model, 'fire_prediction_xgb_model.pkl')


# Visualization: Heat Maps
# Load the saved model
model = joblib.load('fire_prediction_xgb_model.pkl')

# Predict fire occurrences on the full dataset
y_pred_full = model.predict(data_features)

# Add predictions to the original dataset
data['fire_prediction'] = y_pred_full

# Filter out only the rows where fire is predicted
fire_data = data[data['fire_prediction'] == 1]

# Create a list of locations where fire is predicted
fire_locations = list(zip(fire_data['fire_location_latitude'], fire_data['fire_location_longitude']))

# Initialize a Folium map centered around the median location of fire predictions
map_center = [fire_data['fire_location_latitude'].median(), fire_data['fire_location_longitude'].median()]
m = folium.Map(location=map_center, zoom_start=9)

# Add the heatmap layer to the map
HeatMap(fire_locations).add_to(m)

# Save the map to an HTML file
m.save('fire_prediction_heatmap.html')