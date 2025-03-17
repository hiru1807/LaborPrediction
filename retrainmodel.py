import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# ✅ Step 1: Define the dataset (Modify this with real data)
X = np.array([
    [500, 3], [1000, 5], [1500, 7], [2000, 10], [2500, 12],
    [3000, 15], [3500, 18], [4000, 20], [4500, 22], [5000, 25]
])  # Example input: [square_feet, duration]

y = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])  # Example output (Modify as needed)

# ✅ Step 2: Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)  # 100 trees
model.fit(X, y)

# ✅ Step 3: Save the trained model to 'labor_model.dat'
with open('labor_model.dat', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model retrained and saved successfully as 'labor_model.dat'")
