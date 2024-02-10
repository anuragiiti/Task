import streamlit as st
import numpy as np
import pandas as pd


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# TASK-1: K-MEANS CLUSTERING

# Load datasets
train_dataset = pd.read_excel('train.xlsx')
test_dataset = pd.read_excel('test.xlsx')

# KMeans clustering
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(train_dataset.iloc[:, :-1].values)
    wcss.append(kmeans.inertia_)

# Number of clusters found is 4
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
kmeans.fit_predict(train_dataset.iloc[:, :-1].values)

# User input
user_data_point_str = st.sidebar.text_input("Enter data point (comma-separated):", "")
if user_data_point_str:
    user_data_point = list(map(float, user_data_point_str.split(',')))
    user_data_point = [user_data_point]

    # Predict cluster
    predicted_cluster = kmeans.predict(user_data_point)

    # Display result
    st.write(f"Task 1: The data point {user_data_point} belongs to Cluster {predicted_cluster[0]}")

# TASK-2: USING FEATURE SCALING

# Load datasets
X_train_fs = StandardScaler().fit_transform(train_dataset.iloc[:, :-1].values)
X_test_fs = StandardScaler().fit_transform(test_dataset.iloc[:, :].values)

# SVM model
classifier_svm = SVC(kernel='rbf', random_state=0)
classifier_svm.fit(X_train_fs, train_dataset.iloc[:, -1].values)

# Naive Bayes model
classifier_nb = GaussianNB()
classifier_nb.fit(X_train_fs, train_dataset.iloc[:, -1].values)

# Random Forest model
classifier_rf = RandomForestClassifier(n_estimators=30, criterion='entropy', random_state=0)
classifier_rf.fit(X_train_fs, train_dataset.iloc[:, -1].values)



# User input for task-2
user_data_point_str_2 = st.sidebar.text_input("Enter data point for Task-2 (comma-separated):", "")
if user_data_point_str_2:
    user_data_point_2 = list(map(float, user_data_point_str_2.split(',')))
    user_data_point_2 = [user_data_point_2]

    # Predictions
    y_pred_svm = classifier_svm.predict([user_data_point_2])
    y_pred_nb = classifier_nb.predict([user_data_point_2])
    y_pred_rf = classifier_rf.predict([user_data_point_2])
   

    # Display results
    st.write(f"Task 2:")
    st.write(f"SVM Prediction: {y_pred_svm[0]}")
    st.write(f"Naive Bayes Prediction: {y_pred_nb[0]}")
    st.write(f"Random Forest Prediction: {y_pred_rf[0]}")
   

# TASK-3:

# Load dataset
dataset = pd.read_excel('rawdata.xlsx')
dataset['datetime'] = pd.to_datetime(dataset['date'])

# Dropping unnecessary columns
dataset = dataset.drop(['date', 'time'], axis=1)

# Filtering Rows With 'placed' Activity For Inside And Outside
inside_placed = dataset[(dataset['activity'] == 'placed') & (dataset['position'].str.lower() == 'inside')]
outside_placed = dataset[(dataset['activity'] == 'placed') & (dataset['position'].str.lower() == 'outside')]

# Calculating Date-Wise Total Duration For Inside And Outside
inside_duration = inside_placed.groupby(inside_placed['datetime'].dt.date)['number'].count()
outside_duration = outside_placed.groupby(outside_placed['datetime'].dt.date)['number'].count()

# Filtering Rows With 'picked' Activity For Inside And Outside
inside_picked = dataset[(dataset['activity'] == 'picked') & (dataset['position'].str.lower() == 'inside')]
outside_picked = dataset[(dataset['activity'] == 'picked') & (dataset['position'].str.lower() == 'outside')]

# Calculating Date-Wise Number Of Picking Activities For Inside And Outside
inside_picking_count = inside_picked.groupby(inside_picked['datetime'].dt.date)['number'].count()
outside_picking_count = outside_picked.groupby(outside_picked['datetime'].dt.date)['number'].count()

# Display results for Task 3
st.write("Task 3:")
st.write("Date-wise Total Duration for Inside:")
st.write(inside_duration)

st.write("Date-wise Total Duration for Outside:")
st.write(outside_duration)

st.write("Date-wise Number of Picking Activities for Inside:")
st.write(inside_picking_count)

st.write("Date-wise Number of Picking Activities for Outside:")
st.write(outside_picking_count)

