import pandas as pd
from scipy.io import arff
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Compare PCA and t-SNE methods by visualizing Bike Sharing Rental dataset. Explore how
#the different features are shown in the DR components. Build a simple prediction model (for
#example, MLP or Random Forest) to predict the count of total rental bikes and compare the
#performance of the model with the different DR techniques.

# Load ARFF file and convert to DataFrame
arff_file = arff.loadarff('dataset.arff')
df = pd.DataFrame(arff_file[0])
print(df.head())

#standardize the data

features = df.select_dtypes(include=['float64', 'int64']).values
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

#PCA

pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_scaled)


print("Explained variance ratio:", pca.explained_variance_ratio_)

#t-SNE
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(features_scaled)


#plot both PCA and t-SNE results side by side for comparison
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].scatter(pca_result[:, 0], pca_result[:, 1])
ax[0].set_title("PCA projection")
ax[0].set_xlabel("PC1")
ax[0].set_ylabel("PC2")
ax[1].scatter(tsne_result[:, 0], tsne_result[:, 1])
ax[1].set_title("t-SNE projection")
ax[1].set_xlabel("t-SNE1")
ax[1].set_ylabel("t-SNE2")
plt.show()


#Prediction model using Random Forest

X_train, X_test, y_train, y_test = train_test_split(features_scaled, df['count'].values, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error without DR:", mse)
#plot predicted vs actual
plt.scatter(y_test, y_pred)
plt.xlabel("Actual count")
plt.ylabel("Predicted count")
plt.title("Random Forest Predictions without DR")
plt.show()

#Different DR techniques
