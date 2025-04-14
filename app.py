import streamlit as st
import joblib
import json
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model files
@st.cache_resource
def load_model():
    model = joblib.load('kmeans_model.joblib')
    with open('sentence_transformer_name.txt', 'r') as f:
        transformer_name = f.read().strip()
    transformer = SentenceTransformer(transformer_name)
    with open('cluster_info.json', 'r') as f:
        cluster_info = json.load(f)
    return model, transformer, cluster_info

model, transformer, cluster_info = load_model()

# Streamlit interface
st.title("🧠 Product Category Clustering")

user_input = st.text_area("Enter a product description:")
if st.button("Predict Cluster") and user_input:
    embedding = transformer.encode([user_input])
    cluster_id = model.predict(embedding)[0]
    cluster_label = cluster_info.get(str(cluster_id), "Unknown")
    
    st.markdown(f"**Predicted Cluster ID:** `{cluster_id}`")
    st.markdown(f"**Cluster Label:** `{cluster_label}`")
