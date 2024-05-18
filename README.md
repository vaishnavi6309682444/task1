import zipfile
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Path to the zip file
zip_path = 'C:/Users/DELL/Downloads/kc_house_data.csv.zip'

# Path to extract the file (ensure this directory exists or create it)
unzip_path = 'C:/Users/DELL/Downloads/kc_house_data/'

# Create the directory if it doesn't exist
if not os.path.exists(unzip_path):
    os.makedirs(unzip_path)

# Extract the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(unzip_path)

# Path to the CSV file inside the extracted folder
csv_path = os.path.join(unzip_path, 'kc_house_data.csv')

# Load the dataset
housing_data = pd.read_csv(csv_path)

# Display the first few rows of the dataset to ensure it loaded correctly
print(housing_data.head())

# Check for missing values
missing_values = housing_data.isnull().sum()
print(missing_values)

# Select relevant features
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront',
            'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built',
            'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']
target = 'price'

# Split the data into training and testing sets
X = housing_data[features]
y = housing_data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5  # Calculate RMSE from MSE
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R²: {r2}')
