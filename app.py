import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Streamlit app setup
st.title("Air Quality Monitoring Project")

# Generate larger sample data (1000 entries)
dates = [pd.to_datetime('2025-04-01') + pd.Timedelta(days=i) for i in range(1000)]
data = {
    "Date": dates,
    "PM2.5": np.random.randint(20, 150, size=1000),
    "PM10": np.random.randint(40, 200, size=1000),
    "NO2": np.random.randint(10, 60, size=1000),
    "SO2": np.random.randint(5, 30, size=1000),
    "CO": np.round(np.random.uniform(0.5, 2.5, size=1000), 2),
    "O3": np.random.randint(20, 80, size=1000),
    "AQI": np.random.randint(30, 450, size=1000)
}

df_sample = pd.DataFrame(data)

# Add AQI Category
def categorize_aqi(aqi):
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Satisfactory'
    elif aqi <= 200:
        return 'Moderate'
    elif aqi <= 300:
        return 'Poor'
    elif aqi <= 400:
        return 'Very Poor'
    else:
        return 'Severe'

df_sample["AQI_Category"] = df_sample["AQI"].apply(categorize_aqi)

# Select only numeric columns for heatmap
numeric_columns = df_sample[["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI"]]

# AQI Distribution Heatmap
st.subheader("AQI Distribution Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(numeric_columns.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
st.pyplot(fig)

# Category Count
category_counts = df_sample["AQI_Category"].value_counts()
st.subheader("Category Counts")
st.write(category_counts)

# Clustering (KMeans)
features = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
kmeans = KMeans(n_clusters=3, random_state=42)
df_sample["Cluster"] = kmeans.fit_predict(df_sample[features])

# Visualize the clustering result
st.subheader("Clustering of PM2.5 vs PM10")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=df_sample['PM2.5'], y=df_sample['PM10'], hue=df_sample['Cluster'], palette="deep", ax=ax)
st.pyplot(fig)

# Classification (Decision Tree)
# Prepare data for classification
X = df_sample[features]
y = df_sample["AQI_Category"]

# Convert categorical target variable into numeric
y = pd.factorize(y)[0]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate the classifier
y_pred = clf.predict(X_test)

# Classification Report and Confusion Matrix
st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

st.subheader("Confusion Matrix")
st.write(confusion_matrix(y_test, y_pred))
