import streamlit as st
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

st.title("🔧 Predictive Maintenance Dashboard")

# Load data
base_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_path, "output", "part-00000")

st.subheader("📂 Loading Hadoop Output...")

df = pd.read_csv(file_path, sep="\t", header=None)
df.columns = ["engine_id", "data"]

# Split data
data_split = df["data"].str.split(",", expand=True)
data_split = data_split.apply(pd.to_numeric, errors='coerce')

df = pd.concat([df["engine_id"], data_split], axis=1)

# Rename columns
num_cols = df.shape[1]
cols = ["engine_id", "cycle"] + [f"sensor_{i}" for i in range(1, num_cols - 1)]
df.columns = cols

st.subheader("📊 Data Preview")
st.dataframe(df.head())

# RUL calculation
rul = df.groupby("engine_id")["cycle"].max().reset_index()
rul.columns = ["engine_id", "max_cycle"]

df = df.merge(rul, on="engine_id")
df["RUL"] = df["max_cycle"] - df["cycle"]

# Features
X = df.drop(["engine_id", "cycle", "max_cycle", "RUL"], axis=1)
y = df["RUL"]

# Train model
st.subheader("🤖 Training Model...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

st.subheader("📈 Model Performance")
st.write(f"Mean Absolute Error (MAE): {mae:.2f}")

# Visualization
st.subheader("📉 Prediction vs Actual")
chart_df = pd.DataFrame({
    "Actual": y_test.values[:100],
    "Predicted": y_pred[:100]
})
st.line_chart(chart_df)

st.success("✅ Dashboard Ready!")