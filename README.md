# 🔧 Predictive Maintenance using Hadoop & Machine Learning

## 📌 Overview
This project predicts the Remaining Useful Life (RUL) of aircraft engines using NASA CMAPSS dataset.

## ⚙️ Tech Stack
- Apache Hadoop (MapReduce)
- Python
- Scikit-learn
- Pandas
- Streamlit

## 📊 Workflow
Raw Data → Hadoop Processing → ML Model → Prediction Dashboard

## 📈 Result
- MAE: 37.64 cycles

## 🚀 How to Run
1. Run Hadoop MapReduce job
2. Move output to local folder
3. Run:
   python train_model.py
4. Run dashboard:
   python -m streamlit run dashboard.py
