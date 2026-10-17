
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load the housing dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# Prepare features and target
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("SmartPrice-AI Model Results")
print("-" * 35)
print(f"Mean Absolute Error: {mae:.3f}")
print(f"R² Score: {r2:.3f}")
