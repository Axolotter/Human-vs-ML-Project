import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from human_algorithm.human_classifier import human_sort
from data.fetch_data import get_data
from sklearn.model_selection import train_test_split


df, target_name = get_data()

train_df, test_df = train_test_split(
    df,
    test_size=.9,
    random_state=44,
    stratify=df[target_name]
)

# def test_sort(surg, acc):
#     if acc==1:
#         return "O"
#     else:
#         return "N"
    

test_df['human_prediction'] = None
for index, row in test_df.iterrows():
    surg = row['surgical_intervention']
    acc = row['accident']
    smoke = row['smoking']
    test_df.loc[index, 'human_prediction'] = human_sort(acc, surg, smoke)

# print(test_df['human_prediction'])

test_df['correct'] = test_df['human_prediction'] == test_df[target_name]
accuracy = (test_df['human_prediction'] == test_df[target_name]).mean()
print(f"Human classifier accuracy: {accuracy:.2%}")


# Here we print the confusion matrix to see how well the human classifier performed on the test-data subset.
conf_matrix = pd.crosstab(
    test_df[target_name],
    test_df['human_prediction'],
    rownames=['Actual'],
    colnames=['Predicted']
)
print(conf_matrix)

# # Finally, we print one example of a failure case where the human classifier got the prediction wrong.
# failure_row = test_df[test_df['human_prediction'] != test_df[target_name]].iloc[0]
# print("\nFAILURE EXAMPLE")
# print(failure_row[['sepal width', 'petal width', target_name, 'human_prediction']])


# Print a scatter plot showing correct vs incorrect predictions.
os.makedirs("human_algorithm/plots", exist_ok=True)

target_colors = list(map(lambda s: 'g' if s==True else 'r', test_df['correct']))
# print(target_colors)
# print(test_df['correct'])

sns.set_style("whitegrid", {'axes.grid' : False})

fig = plt.figure(figsize=(6,6))

ax = fig.add_subplot(111, projection='3d')

x = 'accident'
y = 'surgical_intervention'
z = 'smoking'
ax.scatter(x, y, z, c=target_colors, data=test_df, marker='o')
# cmap='RdYlGn_r'
#'accident', 'surgical_intervention', 'smoking'
ax.set_xlabel("accident")
ax.set_ylabel("surgical_intervention")
ax.set_zlabel("smoking")
# plt.legend(title='Prediction Correct')
plt.title(f"Fertility Diagnosis: accident vs surgical_intervention vs smoking")

plt.savefig(f'human_algorithm/plots/human_model_training_results.png', dpi=150)

# plt.figure(figsize=(8, 6))
# sns.scatterplot(
#     data=test_df,
#     x='sepal width',
#     y='petal width',
#     hue='correct',
#     style='correct',
#     s=100,
#     palette={True: 'green', False: 'red'}
# )

plt.close()