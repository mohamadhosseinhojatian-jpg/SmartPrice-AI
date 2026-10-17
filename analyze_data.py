import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Load dataset
print("Loading California Housing dataset...")
data = fetch_california_housing(as_frame=True)
df = data.frame

# Basic analysis
print("\n--- Dataset Overview ---")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print("\nFirst 5 rows:")
print(df.head())

print("\n--- Statistical Summary ---")
print(df.describe().round(2))

print("\n--- Missing Values ---")
print(df.isnull().sum())

# Correlation with house value
correlation = df.corr()["MedHouseVal"].sort_values(ascending=False)

print("\n--- Correlation with House Value ---")
print(correlation)

# Create correlation chart
correlation.drop("MedHouseVal").sort_values().plot(
    kind="barh",
    title="Feature Correlation with House Value",
    figsize=(10, 6)
)

plt.xlabel("Correlation")
plt.tight_layout()
plt.savefig("correlation_chart.png", dpi=300)
print("\nChart saved as correlation_chart.png")