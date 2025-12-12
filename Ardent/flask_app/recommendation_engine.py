import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from surprise import Dataset, Reader, KNNBasic
import matplotlib.pyplot as plt
import os

# Get the absolute path to the CSV file
base_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(base_dir, "student-mat.csv"), sep=";")

label = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = label.fit_transform(df[col])

cluster_features = ["studytime","goout","freetime","absences",
                    "traveltime","failures","G1","G2","G3"]

X_cluster = df[cluster_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

ratings = pd.read_csv(os.path.join(base_dir, "student_material_ratings.csv"))
reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(ratings[["student_id", "material_id", "rating"]], reader)
trainset = data.build_full_trainset()
algo = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
algo.fit(trainset)

def cf_recommend(user_id):
    all_items = ratings["material_id"].unique()
    rated = ratings[ratings["student_id"] == user_id]["material_id"].unique()
    to_predict = [m for m in all_items if m not in rated]

    predictions = []
    for m in to_predict:
        pred = algo.predict(user_id, m).est
        predictions.append((m, pred))

    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:5]

def cluster_recommend(cluster_id):
    if cluster_id == 0:
        return ["Basic Videos","Intro Notes","Beginner Worksheets"]
    elif cluster_id == 1:
        return ["Practice Quizzes","Summary Notes","Moderate Exercises"]
    else:
        return ["Advanced Problems","Projects","Challenge Assignments"]

def get_recommendations(student_id):
    cf_items = cf_recommend(student_id)
    cluster_items = cluster_recommend(df.loc[student_id, "Cluster"])
    return {"cf": cf_items, "cluster": cluster_items}

def get_progress_chart(student_id):
    grades = df.loc[student_id, ["G1","G2","G3"]].values

    plt.figure(figsize=(6,4))
    plt.plot([1,2,3], grades, marker='o')
    plt.xticks([1,2,3], ["G1","G2","G3"])
    plt.title(f"Progress Chart for Student {student_id}")
    plt.xlabel("Exam Period")
    plt.ylabel("Grade")

    path = f"static/progress_{student_id}.png"
    plt.savefig(path)
    plt.close()
    return path

def predict_performance(student_id):
    X = df.drop("G3", axis=1)
    y = df["G3"]

    scaler2 = StandardScaler()
    X_scaled = scaler2.fit_transform(X)

    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_scaled, y)

    student_data = X_scaled[student_id].reshape(1, -1)
    predicted = rf.predict(student_data)[0]
    return round(predicted, 2)
