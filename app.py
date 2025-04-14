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
    # Inference
    embedding = transformer.encode([user_input])
    cluster_id = model.predict(embedding)[0]
    cluster_label = cluster_info.get(str(cluster_id), "Unknown")

    # Display results
    st.subheader("🔍 Prediction Results")
    st.markdown(f"**Predicted Cluster ID:** `{cluster_id}`")

    if isinstance(cluster_label, dict):
        size = cluster_label.get("size", "N/A")
        top_terms = cluster_label.get("top_terms", {})
        sorted_terms = sorted(top_terms.items(), key=lambda x: x[1], reverse=True)

        st.markdown(f"**Cluster Size:** `{size}`")
        st.markdown("**Top Terms in Cluster:**")
        for term, count in sorted_terms:
            st.markdown(f"- 🟢 **{term}** (_{count}_)")
    else:
        st.markdown(f"**Cluster Label:** `{cluster_label}`")

