# combine collaborative filtering results with clustering data to create personalized content fee


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from surprise import Dataset, Reader, KNNBasic


# ========================================================
# 1. Load Student Dataset
# ========================================================
df = pd.read_csv("student-mat.csv", sep=";")

# Encode all categorical columns (use a fresh encoder per column)
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])


# ========================================================
# 2. K-Means Clustering (Student Behavior Patterns)
# ========================================================
cluster_features = [
    "studytime", "goout", "freetime", "absences",
    "traveltime", "failures", "G1", "G2", "G3"
]

X_cluster = df[cluster_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Create a `student_id` column that maps dataset rows to the rating user ids.
# The `student_material_ratings.csv` uses 1-based student ids, so map rows
# to 1..N to allow joining the two datasets.
df["student_id"] = np.arange(1, len(df) + 1)



# ========================================================
# 3. Collaborative Filtering Dataset (Student-Material Ratings)
# ========================================================
ratings = pd.read_csv("student_material_ratings.csv")

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[["student_id", "material_id", "rating"]], reader)
trainset = data.build_full_trainset()

algo = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
algo.fit(trainset)



# ========================================================
# 4. CF Recommendation Function
# ========================================================
def cf_recommend(user_id, df_rate, algo, top_n=5):

    all_materials = df_rate["material_id"].unique()
    rated = df_rate[df_rate["student_id"] == user_id]["material_id"].unique()

    items_to_predict = [m for m in all_materials if m not in rated]

    results = []
    for mat in items_to_predict:
        pred_score = algo.predict(user_id, mat).est
        results.append((mat, pred_score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]



# ========================================================
# 5. Cluster-Based Learning Content Recommendations
# ========================================================
def cluster_content(cluster_id):

    if cluster_id == 0:
        return [
            ("Basic Video Lessons", 0.6),
            ("Beginner Tutorials", 0.55),
            ("Simple Practice Worksheets", 0.50)
        ]

    elif cluster_id == 1:
        return [
            ("Practice Quizzes", 0.65),
            ("Revision Notes", 0.60),
            ("Medium-Level Exercises", 0.55)
        ]

    elif cluster_id == 2:
        return [
            ("Advanced Problem Sets", 0.75),
            ("Challenge Assignments", 0.70),
            ("Project-Based Learning Content", 0.68)
        ]

    return []



# ========================================================
# 6. FINAL PERSONALIZED CONTENT FEED (CF + CLUSTER)
# ========================================================
def personalized_feed(user_id, df, ratings, algo):
    
    # ---------------- CF RECOMMENDATIONS ----------------
    cf_items = cf_recommend(user_id, ratings, algo, top_n=5)

    # ---------------- CLUSTER RECOMMENDATIONS ----------------
    # Look up the user's cluster by matching the `student_id` column.
    user_row = df[df["student_id"] == user_id]
    if user_row.empty:
        raise ValueError(f"User id {user_id} not found in student dataset")
    user_cluster = int(user_row["Cluster"].iloc[0])
    cluster_items = cluster_content(user_cluster)

    # Weighting system
    # CF has stronger personalization -> weight 0.7
    # Cluster recs -> weight 0.3
    
    feed = []

    # Add CF items with weights
    for mat, score in cf_items:
        feed.append((f"Material {mat}", score * 0.7))

    # Add cluster items with weights
    for item, score in cluster_items:
        feed.append((item, score * 0.3))

    # Sort final feed
    feed.sort(key=lambda x: x[1], reverse=True)

    return cf_items, cluster_items, feed



# ========================================================
# 7. RUN FINAL FEED FOR A SPECIFIC STUDENT
# ========================================================
user_id = 5
cf_res, cluster_res, final_feed = personalized_feed(user_id, df, ratings, algo)

print("\n==============================")
print(f"PERSONALIZED FEED FOR STUDENT {user_id}")
print("==============================\n")

print("🔹 Collaborative Filtering Recommendations:\n")
for item, score in cf_res:
    print(f"  ➤ Material {item} (Predicted Rating: {score:.2f})")

print("\n🔹 Cluster-Based Recommendations:\n")
for item, score in cluster_res:
    print(f"  ➤ {item} (Cluster Weight: {score})")

print("\n🔹 FINAL PERSONALIZED CONTENT FEED:\n")
for item, score in final_feed:
    print(f"  ⭐ {item} → Score: {score:.2f}")
