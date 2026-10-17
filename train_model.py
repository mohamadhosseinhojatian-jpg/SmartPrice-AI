
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

print("Loading dataset...")
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1
    )
}

best_model = None
best_score = float("-inf")
best_name = ""

print("\nModel comparison")
print("-" * 45)

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"{name}:")
    print(f"  MAE: {mae:.3f}")
    print(f"  R²:  {r2:.3f}\n")

    if r2 > best_score:
        best_score = r2
        best_model = model
        best_name = name

joblib.dump(best_model, "house_price_model.pkl")

print("-" * 45)
print(f"Best model: {best_name}")
print(f"Best R² score: {best_score:.3f}")
print("Best model saved successfully!")
