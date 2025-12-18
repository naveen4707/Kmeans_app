import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="Student Performance Clustering", page_icon="🎓", layout="centered")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .cluster-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        background-color: #ffffff;
        border: 2px solid #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_model():
    with open('Kmeans.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('train_data.pkl', 'rb') as f:
        data = pickle.load(f)
    return model, data

try:
    kmeans, X_train = load_model()
except:
    st.error("Missing model files! Please run the generation script first.")
    st.stop()

# --- UI HEADER ---
st.title("🎓 Student Marks Clustering")
st.write("Enter student marks to identify which academic performance cluster they belong to.")

# --- INPUT SECTION ---
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        maths = st.number_input("Maths Marks", min_value=0, max_value=100, value=50)
    with col2:
        science = st.number_input("Science Marks", min_value=0, max_value=100, value=50)

# --- PREDICTION ---
if st.button("Analyze Cluster"):
    user_data = np.array([[maths, science]])
    cluster = kmeans.predict(user_data)[0]
    
    st.markdown(f"""
        <div class="cluster-box">
            <h3>Predicted Cluster: <span style="color:#4CAF50;">{cluster}</span></h3>
        </div>
    """, unsafe_allow_html=True)

    # --- VISUALIZATION ---
    st.divider()
    st.subheader("📊 Visualization")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot existing training data
    labels = kmeans.labels_
    scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=labels, cmap='viridis', label='Existing Students', alpha=0.6)
    
    # Plot Cluster Centers
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
               marker='X', s=200, c='red', label='Centroids')
    
    # Plot User Input
    ax.scatter(maths, science, marker='o', s=300, c='orange', edgecolors='black', label='Your Input')
    
    ax.set_xlabel("Maths Marks")
    ax.set_ylabel("Science Marks")
    ax.legend()
    st.pyplot(fig)

st.sidebar.info("This app uses K-Means Clustering (K=4) to categorize students based on their performance in Maths and Science.")
