from sklearn.feature_extraction.text import TfidfVectorizer

import workshop1 as rnn_module  # Using your attached workshop1.py for RNN implementation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load the CSV file
df = pd.read_csv("df_file.csv")

# Extract text and labels columns
texts = df['Text'].astype(str).tolist()
labels = df['Label'].astype(int).tolist()
num_classes = len(set(labels))


print(num_classes)

def text_to_numpy_sequences(texts, seq_len=None):
    vect = TfidfVectorizer(analyzer='char', ngram_range=(1,1))
    X_mat = vect.fit_transform(texts).toarray().astype(np.float32)
    if seq_len is None:
        seq_len = X_mat.shape[1]
    # truncate or pad to seq_len
    if X_mat.shape[1] >= seq_len:
        X_mat = X_mat[:, :seq_len]
    else:
        pad = np.zeros((X_mat.shape[0], seq_len - X_mat.shape[1]), dtype=np.float32)
        X_mat = np.hstack([X_mat, pad])
    X = np.expand_dims(X_mat, axis=2)  # shape (n_samples, seq_len, 1)
    feature2idx = {feat: i+1 for i, feat in enumerate(vect.get_feature_names_out())}
    return X, feature2idx

def text_to_tfidf_sequences(texts, seq_len=None):
    vect = TfidfVectorizer(analyzer='char', ngram_range=(1,1))
    X_mat = vect.fit_transform(texts).toarray().astype(np.float32)
    if seq_len is None:
        seq_len = X_mat.shape[1]
    # truncate or pad to seq_len
    if X_mat.shape[1] >= seq_len:
        X_mat = X_mat[:, :seq_len]
    else:
        pad = np.zeros((X_mat.shape[0], seq_len - X_mat.shape[1]), dtype=np.float32)
        X_mat = np.hstack([X_mat, pad])
    X = np.expand_dims(X_mat, axis=2)  # shape (n_samples, seq_len, 1)
    feature2idx = {feat: i+1 for i, feat in enumerate(vect.get_feature_names_out())}
    return X, feature2idx


# Convert labels to one-hot encoded vectors

def labels_to_onehot(labels, num_classes):
    labels = np.asarray(labels)
    uniques, inv = np.unique(labels, return_inverse=True)
    if num_classes is None:
        num_classes = len(uniques)
    y = np.eye(num_classes, dtype=np.float32)[inv]
    return y

seq_len = 500
X, feature2idx = text_to_numpy_sequences(texts, seq_len)
y = labels_to_onehot(labels, num_classes)

# Split the dataset into training, validation, and test sets

# Shuffle the data
n_samples = X.shape[0]

indices = np.random.permutation(n_samples)

split = int(0.8 * n_samples)
train_idx, test_idx = indices[:split], indices[split:]
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

# Instantiate the RNN from workshop1.py module
rnn = rnn_module.RNN(input_size=1, hidden_size=64, output_size=num_classes, lr=5e-4)
#(self, input_size, hidden_size, output_size, lr=1e-3, seed=1)

# Train the RNN
losses = rnn.train(X_train, y_train, epochs=30,batch_size=32, verbose=True)



# Predict on the test set
_, y_pred_test = rnn.forward(X_test)
y_pred_labels = np.argmax(y_pred_test, axis=1)
true_labels = np.argmax(y_test, axis=1)
accuracy = np.mean(y_pred_labels == true_labels)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Plot training loss curve
plt.plot(losses)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

#Accuracy per category
for class_idx in range(num_classes):
    class_mask = (true_labels == class_idx)
    class_accuracy = np.mean(y_pred_labels[class_mask] == true_labels[class_mask])
    print(f"Accuracy for class {class_idx}: {class_accuracy*100:.2f}%")

#Plot predictions vs true labels
plt.figure(figsize=(12,6))
plt.plot(true_labels, label='True Labels', marker='o')
plt.plot(y_pred_labels, label='Predicted Labels', marker='x')
plt.title("True vs Predicted Labels ")
plt.xlabel("Sample Index")
plt.ylabel("Class Label")
plt.legend()
plt.show()