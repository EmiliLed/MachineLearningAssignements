import pandas as pd
from scipy.io import arff
#Visualize MNIST-784 handwritten digits dataset with SOM and discuss what you can learn
#from the visualization.

arff_file = arff.loadarff('mnist_784.arff')

#SOM implementation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from minisom import MiniSom
# Convert to DataFrame
df = pd.DataFrame(arff_file[0])
# Separate features and labels
data = df.drop('class', axis=1).values
labels = df['class'].values.astype(int)
# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
# Initialize and train SOM
som_size = 20
som = MiniSom(som_size, som_size, data_scaled.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(data_scaled)
som.train(data_scaled, 1000, verbose=True)
# Create a mapping of each neuron to the labels
from collections import defaultdict
label_map = defaultdict(list)
for i, x in enumerate(data_scaled):
    w = som.winner(x)
    label_map[w].append(labels[i])

# Plot U-matrix (distance map)
plt.figure(figsize=(8, 8))
plt.title('SOM U-matrix (Distance Map)')
plt.pcolor(som.distance_map().T, cmap='coolwarm')  # U-matrix shows distances between neurons
plt.colorbar()
plt.show()

# Plot the SOM with labels
plt.figure(figsize=(10, 10))
for i in range(som_size):
    for j in range(som_size):
        plt.subplot(som_size, som_size, i * som_size + j + 1)
        plt.axis('off')
        if (i, j) in label_map:
            counts = np.bincount(label_map[(i, j)])
            plt.text(0.5, 0.5, str(np.argmax(counts)), fontsize=12, ha='center', va='center')
plt.suptitle('SOM Visualization of MNIST-784 Handwritten Digits')
plt.show()
# The visualization shows how different handwritten digits are clustered in the SOM grid.
