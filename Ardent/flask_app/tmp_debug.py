import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from surprise import Dataset, Reader, KNNBasic

print('START: importing done')

df = pd.read_csv("student-mat.csv", sep=';')
print('Loaded student-mat:', df.shape)

label = LabelEncoder()
for i, col in enumerate(df.select_dtypes(include=['object']).columns):
    df[col] = label.fit_transform(df[col])
    if i<5:
        print('Encoded col:', col)

cluster_features = [
    "studytime", "goout", "freetime", "absences",
    "traveltime", "failures", "G1", "G2", "G3"
]
X = df[cluster_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print('Scaled features shape:', X_scaled.shape)

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
print('KMeans done. Cluster value counts:\n', df['Cluster'].value_counts())

ratings = pd.read_csv('student_material_ratings.csv')
print('Loaded ratings:', ratings.shape)

reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(ratings[['student_id','material_id','rating']], reader)
print('Surprise Dataset created')

trainset = data.build_full_trainset()
print('Trainset built', type(trainset))

sim_options = {'name':'cosine','user_based':True}
algo = KNNBasic(sim_options=sim_options)
print('Fitting algo...')
algo.fit(trainset)
print('Algo fit complete')

# simple predict
pred = algo.predict(5, 1)
print('Prediction object for user 5 material 1:', pred)

print('END: debug script finished')
