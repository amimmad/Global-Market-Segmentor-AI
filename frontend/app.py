import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os
from sklearn.ensemble import IsolationForest, RandomForestClassifier

# Path configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.processor import DataProcessor
from src.model import ClusterModel

st.set_page_config(page_title="World Insight Pro", layout="wide")

# High-End UI Styling
st.markdown("""
    <style>
    .reportview-container { background: #0f172a; }
    .stMetric { border-left: 5px solid #3b82f6; background: #1e293b; padding: 15px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 Global Market-Segmentor: Advanced AI Analytics")

uploaded_file = "data/CIA_Country_Facts.csv"

if os.path.exists(uploaded_file):
    # Data Engine
    df_raw = pd.read_csv(uploaded_file)
    processor = DataProcessor(n_components=3)
    pca_df, scaled_data = processor.process_data(df_raw)

    # 1. ANOMALY DETECTION (The "Cool" Factor)
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    anomalies = iso_forest.fit_predict(scaled_data)
    pca_df['Is_Anomaly'] = ["Anomaly" if x == -1 else "Normal" for x in anomalies]

    # Sidebar
    st.sidebar.header("Intelligence Settings")
    k = st.sidebar.slider("Clustering Granularity (K)", 2, 10, 5)

    # Clustering
    model = ClusterModel(n_clusters=k)
    clusters = model.fit_predict(scaled_data)
    pca_df['Cluster'] = clusters.astype(str)
    pca_df['Country'] = df_raw['Country']

    # Visual Tabs
    t1, t2, t3 = st.tabs(["🌐 3D Map", "🧠 AI Feature Importance", "🚨 Anomaly Detection"])

    with t1:
        fig_3d = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', color='Cluster',
                               hover_data=['Country'], template='plotly_dark', height=700)
        st.plotly_chart(fig_3d, use_container_width=True)

    with t2:
        st.subheader("Which features drive these clusters?")
        # Using Random Forest to find feature importance
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(scaled_data, clusters)

        # Get feature names from processor
        feat_names = processor.get_feature_names(processor.clean_dataframe(df_raw))
        importance_df = pd.DataFrame({'Feature': feat_names, 'Importance': rf.feature_importances_})
        importance_df = importance_df.sort_values('Importance', ascending=False)

        fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                         title="AI Decision Drivers", template="plotly_dark")
        st.plotly_chart(fig_imp, use_container_width=True)

    with t3:
        st.subheader("Global Outliers (Anomalies)")
        st.write("These countries don't follow the global pattern:")
        anomaly_list = pca_df[pca_df['Is_Anomaly'] == "Anomaly"]['Country'].tolist()
        st.error(", ".join(anomaly_list))

        fig_anom = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', color='Is_Anomaly',
                                 symbol='Is_Anomaly', color_discrete_map={'Anomaly': 'red', 'Normal': 'gray'},
                                 template='plotly_dark')
        st.plotly_chart(fig_anom, use_container_width=True)

    # 4. Recommendation System (Actionable Insight)
    st.markdown("---")
    st.subheader("🤝 Country Similarity Finder")
    target = st.selectbox("Pick a country:", pca_df['Country'].unique())
    c_id = pca_df[pca_df['Country'] == target]['Cluster'].values[0]
    peers = pca_df[pca_df['Cluster'] == c_id]['Country'].tolist()
    if target in peers: peers.remove(target)
    st.info(f"Countries with similar DNA to {target}:  \n" + ", ".join(peers[:15]))

else:
    st.error("File not found!")
