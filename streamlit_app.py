# app.py
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# Load the training data
train_dataset = pd.read_excel('train.xlsx')
X_train = train_dataset.iloc[:, :-1].values

# Train the K-Means model
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
kmeans.fit(X_train)

# Function to predict cluster for user input
def predict_cluster(user_input):
    user_data_point = list(map(float, user_input.split(',')))
    user_data_point = [user_data_point]
    predicted_cluster = kmeans.predict(user_data_point)
    return predicted_cluster[0]

# Streamlit UI
def main():
    st.title('K-Means Clustering Model Deployment')

    # Sidebar for user input
    st.sidebar.header('User Input')
    user_input_str = st.sidebar.text_input('Enter data point (comma-separated):', "-77,-74,-71,-76,-65,-63,-66,-52,-55,-75,-72,-75,-74,-61,-64,-63,-53,-63")
    if st.sidebar.button('Predict Cluster'):
        predicted_cluster = predict_cluster(user_input_str)
        st.sidebar.success(f'The data point belongs to Cluster {predicted_cluster}')

   



if __name__ == '__main__':
    main()
