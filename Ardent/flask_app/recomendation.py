# ================================================================
#     FULL COMBINED RECOMMENDATION ENGINE (K-Means + CF HYBRID)
# ================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from surprise import Dataset, Reader, KNNBasic


# ================================================================
# 1. LOAD STUDENT DATA
# ================================================================
df = pd.read_csv("student-mat.csv", sep=";")
# df = pd.read_csv("student-por.csv", sep=";")   # optional

# Ensure there is a `student_id` column mapping each row to a stable id
# (this lets cluster_recommend look up clusters by student id from ratings)
if "student_id" not in df.columns:
    df["student_id"] = np.arange(1, len(df) + 1)

# Encode categorical columns
for col in df.select_dtypes(include=['object']).columns:
    # use a fresh LabelEncoder per column to avoid shared state
    df[col] = LabelEncoder().fit_transform(df[col])


# ================================================================
# 2. K-MEANS CLUSTERING (STUDENT BEHAVIOR GROUPING)
# ================================================================
cluster_features = [
    "studytime", "goout", "freetime", "absences",
    "traveltime", "failures", "G1", "G2", "G3"
]

X = df[cluster_features]

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train K-Means model
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)



# ================================================================
# 3. LOAD RATING DATA FOR COLLABORATIVE FILTERING
# ================================================================
# student_id, material_id, rating CSV REQUIRED
ratings = pd.read_csv("student_material_ratings.csv")

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[["student_id", "material_id", "rating"]], reader)

trainset = data.build_full_trainset()

sim_options = {
    "name": "cosine",
    "user_based": True
}

algo = KNNBasic(sim_options=sim_options)
algo.fit(trainset)



# ================================================================
# 4. COLLABORATIVE FILTERING RECOMMENDATION FUNCTION
# ================================================================
def cf_recommend(user_id, ratings, algo, top_n=5):
    
    all_materials = ratings["material_id"].unique()
    rated_materials = ratings[ratings["student_id"] == user_id]["material_id"].unique()

    # Materials not yet rated by the user
    materials_to_predict = [m for m in all_materials if m not in rated_materials]

    predictions = []
    for material in materials_to_predict:
        pred = algo.predict(user_id, material).est
        predictions.append((material, pred))

    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:top_n]



# ================================================================
# 5. CLUSTER-BASED CONTENT RECOMMENDER
# ================================================================
def cluster_recommend(user_id, df):
    # If the dataframe contains an explicit `student_id` column, use it.
    # Otherwise assume `user_id` is a 1-based positional id that maps
    # to the dataframe row index (i.e. student 1 -> df.iloc[0]).
    if "student_id" in df.columns:
        row = df[df["student_id"] == user_id]
        if row.empty:
            raise ValueError(f"student_id {user_id} not found in dataframe")
        user_cluster = row.iloc[0]["Cluster"]
    else:
        # treat user_id as 1-based position by default
        idx = user_id - 1 if user_id > 0 else user_id
        if idx < 0 or idx >= len(df):
            raise IndexError(f"user_id {user_id} is out of bounds for dataframe of length {len(df)}")
        user_cluster = df.iloc[idx]["Cluster"]

    if user_cluster == 0:
        return [
            "Basic Video Lessons",
            "Beginner-Level Tutorials",
            "Simple Practice Worksheets"
        ]

    elif user_cluster == 1:
        return [
            "Practice Quizzes",
            "Revision Notes",
            "Moderate-Level Exercises"
        ]

    elif user_cluster == 2:
        return [
            "Advanced Problem Sets",
            "Challenging Assignments",
            "Project-Based Learning Content"
        ]

    else:
        return ["General Learning Materials"]



# ================================================================
# 6. FINAL HYBRID RECOMMENDATION ENGINE
# ================================================================
def hybrid_recommendation(user_id, df, ratings, algo):

    print("\n======================================================")
    print(f"   🔍 RECOMMENDATION ENGINE RESULTS FOR STUDENT {user_id}")
    print("======================================================\n")

    # --------------- Collaborative Filtering ---------------
    cf_results = cf_recommend(user_id, ratings, algo, top_n=5)

    print("📌 TOP COLLABORATIVE FILTERING RECOMMENDATIONS:\n")
    for mat, score in cf_results:
        print(f"   ➤ Material {mat} — Predicted Rating: {score:.2f}")

    # --------------- Cluster-Based Content ---------------
    cl_results = cluster_recommend(user_id, df)

    print("\n📌 CLUSTER-BASED LEARNING MATERIAL SUGGESTIONS:\n")
    for c in cl_results:
        print(f"   • {c}")

    print("\n======================================================\n")

    return cf_results, cl_results



# ================================================================
# 7. RUN THE FINAL HYBRID ENGINE (Example: User 5)
# ================================================================
hybrid_recommendation(user_id=5, df=df, ratings=ratings, algo=algo)
