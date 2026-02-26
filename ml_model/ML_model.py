import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from data.fetch_data import get_data
from sklearn.svm import NuSVC
from sklearn.neural_network import MLPClassifier
from sklearn.svm import OneClassSVM
df, target_name = get_data()

# I selected only the petal length and petal width features for classification.
X = df[['surgical_intervention', 'accident', 'smoking']]
y = df[target_name]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.7,
    random_state=42,
    stratify=y
)

# BEST METHOD SO FAR: NuSVC test size .7 k .2 0 false negatives
# NuSVC range: 0.08-.23 best found: 0.08 with .5 test split
# RandomForest range: best between 1-10
# MLPClassifier range: 1
# KNN: k =1
k = .2
r = 1
# ml = KNeighborsClassifier(n_neighbors=k)
# ml = RandomForestClassifier(n_estimators=k, random_state=r)
# ml = LinearSVC(random_state=r)
ml = NuSVC(nu=k, random_state=r)
# ml = MLPClassifier(random_state=r, max_iter=k)
# ml = OneClassSVM(nu=k)
ml.fit(X_train, y_train)
y_pred = ml.predict(X_test)
y_train_pred = ml.predict(X_train)
# print(y_pred)
# print(y_test)
# print(y_train_pred)
# create confusion matrix
conf_matrix_ml = pd.crosstab(
    y_test,
    y_pred,
    rownames=['Actual'],
    colnames=['Predicted']
)

# compute accuracy on test data
accuracy_ml = (y_pred == y_test).mean()

# display results on test data
print(f"ML classifier accuracy (k={k}): {accuracy_ml:.2%}\n")
print(conf_matrix_ml)

# Add a 'correct' column for the visualization on test data
test_df = X_test.copy()
test_df[target_name] = y_test
test_df['ML_prediction'] = y_pred
test_df['correct'] = test_df['ML_prediction'] == test_df[target_name]

# Add a 'correct' column for the visualization on training data
train_df = X_train.copy()
train_df[target_name] = y_train
train_df['ML_prediction'] = y_train_pred
train_df['correct'] = train_df['ML_prediction'] == train_df[target_name]

# Create a visualization of KNN classifier results
os.makedirs("ml_model/plots", exist_ok=True)

target_colors = list(map(lambda s: 'g' if s=='N' else 'r', df[target_name]))
# print(target_colors)

sns.set_style("whitegrid", {'axes.grid' : False})

fig = plt.figure(figsize=(6,6))

ax = fig.add_subplot(111, projection='3d')

x = "surgical_intervention"
y = "accident"
z = "smoking"
ax.scatter(x, y, z, c=target_colors, data=df, marker='o')
# cmap='RdYlGn_r'
ax.set_xlabel(x)
ax.set_ylabel(y)
ax.set_zlabel(z)

plt.title(f"Fertility Diagnosis: MLPClassifier Model Training Results, k={k}, split=.7")

plt.savefig(f'ml_model/plots/MLP_model_training_results.png', dpi=150)
plt.close()
