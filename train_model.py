import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ✅ STEP 1: Read Hadoop output
df = pd.read_csv("output/part-00000", sep="\t", header=None)
df.columns = ["engine_id", "data"]

# ✅ STEP 2: Split sensor data
data_split = df["data"].str.split(",", expand=True)

# Convert to numeric
data_split = data_split.apply(pd.to_numeric)

# Combine back
df = pd.concat([df["engine_id"], data_split], axis=1)

# Rename columns
num_cols = df.shape[1]
cols = ["engine_id", "cycle"] + [f"sensor_{i}" for i in range(1, num_cols - 1)]
df.columns = cols

# ✅ STEP 3: Calculate RUL
rul = df.groupby("engine_id")["cycle"].max().reset_index()
rul.columns = ["engine_id", "max_cycle"]

df = df.merge(rul, on="engine_id")
df["RUL"] = df["max_cycle"] - df["cycle"]

# ✅ STEP 4: Features & target
X = df.drop(["engine_id", "cycle", "max_cycle", "RUL"], axis=1)
y = df["RUL"]

# ✅ STEP 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ STEP 6: Model
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# ✅ STEP 7: Prediction
y_pred = model.predict(X_test)

# ✅ STEP 8: Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))