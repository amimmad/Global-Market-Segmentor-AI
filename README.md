# 🌍 Market-Segmentor AI: Global Economic Clustering
![Python](https://img.shields.io/badge/Python-3.14+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML--Engine-F7931E.svg)
![Plotly](https://img.shields.io/badge/Plotly-3D--Visualization-440154.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An advanced MLOps project that uses Unsupervised Machine Learning to segment countries based on socio-economic indicators from the CIA World Factbook.

## 🚀 Key Features
- **3D Latent Space Visualization:** PCA-reduced dimensionality for interactive exploration.
- **AI Decision Drivers (XAI):** Uses Random Forest to explain *why* countries are grouped together.
- **Anomaly Detection:** Identifies global outliers using Isolation Forest.
- **Smart Recommender:** Suggests peer countries based on socio-economic proximity.

## 🛠️ Tech Stack
- **Backend:** Python, Scikit-Learn
- **Frontend:** Streamlit
- **Visualization:** Plotly Express
- **Data:** CIA World Factbook

## 📉 Mathematical Logic
The project utilizes **K-Means Clustering** optimized by the **Elbow Method**. High-dimensional data (20+ features) is compressed into a 3D space using **Principal Component Analysis (PCA)** to retain maximum variance while allowing visualization.

## 🖥️ How to Run
1. `pip install -r requirements.txt`
2. `streamlit run frontend/app.py`
