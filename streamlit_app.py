import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# Importing datasets
train_dataset = pd.read_excel('train.xlsx')
X_train = train_dataset.iloc[:, :-1].values
y_train = train_dataset.iloc[:, -1].values

# TASK-1: K-MEANS CLUSTERING
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_train)
    wcss.append(kmeans.inertia_)

# The number of clusters found is 4
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
kmeans.fit_predict(X_train)

# TASK-2: USING FEATURE SCALING
sc = StandardScaler()
X_train_fs = sc.fit_transform(X_train)

# Training the KERNEL SVM model on the Training set
classifier_svm = SVC(kernel='rbf', random_state=0)
classifier_svm.fit(X_train_fs, y_train)

# Training the NAIVE BAYES model on the Training set
classifier_nb = GaussianNB()
classifier_nb.fit(X_train_fs, y_train)

# Training the RANDOM FOREST model on the Training set
classifier_rf = RandomForestClassifier(n_estimators=30, criterion='entropy', random_state=0)
classifier_rf.fit(X_train_fs, y_train)

# TASK-3: PROCESSING DATASET
dataset = pd.read_excel('rawdata.xlsx')
dataset['datetime'] = pd.to_datetime(dataset['date'])
dataset = dataset.drop(['date', 'time'], axis=1)

inside_placed = dataset[(dataset['activity'] == 'placed') & (dataset['position'].str.lower() == 'inside')]
outside_placed = dataset[(dataset['activity'] == 'placed') & (dataset['position'].str.lower() == 'outside')]

inside_duration = inside_placed.groupby(inside_placed['datetime'].dt.date)['number'].count()
outside_duration = outside_placed.groupby(outside_placed['datetime'].dt.date)['number'].count()

inside_picked = dataset[(dataset['activity'] == 'picked') & (dataset['position'].str.lower() == 'inside')]
outside_picked = dataset[(dataset['activity'] == 'picked') & (dataset['position'].str.lower() == 'outside')]

inside_picking_count = inside_picked.groupby(inside_picked['datetime'].dt.date)['number'].count()
outside_picking_count = outside_picked.groupby(outside_picked['datetime'].dt.date)['number'].count()

# Streamlit App
st.title('Streamlit App for Task 1, Task 2, and Task 3')

# User Input Section
user_data_point_str = st.text_input("Enter data point for Task-2 (comma-separated):", "-77,-74,-71,-76,-65,-63,-66,-52,-55,-75,-72,-75,-74,-61,-64,-63,-53,-63")
user_data_point = list(map(float, user_data_point_str.split(',')))
user_data_point = [user_data_point]

# Predictions
predicted_cluster = kmeans.predict(user_data_point)
predicted_svm = classifier_svm.predict(sc.transform(user_data_point))
predicted_nb = classifier_nb.predict(sc.transform(user_data_point))
predicted_rf = classifier_rf.predict(sc.transform(user_data_point))

# Display Predictions
st.header("Predictions for Task-2:")
st.write(f"The data point {user_data_point} belongs to Cluster {predicted_cluster[0]}")

st.header("Predictions for Task-2 using SVM:")
st.write(f"Prediction: {predicted_svm[0]}")

st.header("Predictions for Task-2 using Naive Bayes:")
st.write(f"Prediction: {predicted_nb[0]}")

st.header("Predictions for Task-2 using Random Forest:")
st.write(f"Prediction: {predicted_rf[0]}")

# Display Results for Task-3
st.header("Results for Task-3:")
st.write("Date-wise Total Duration for Inside:")
st.write(inside_duration)

st.write("Date-wise Total Duration for Outside:")
st.write(outside_duration)

st.write("Date-wise Number of Picking Activities for Inside:")
st.write(inside_picking_count)

st.write("Date-wise Number of Picking Activities for Outside:")
st.write(outside_picking_count)

