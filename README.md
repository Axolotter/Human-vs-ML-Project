# Human vs Machine Learning Project

This project challenges you to explore the differences between human-designed algorithms and machine learning models. You will first create a human algorithm (pseudo-code) to classify data based on features, then translate that algorithm into Python. Next, you will train a K-Nearest Neighbors (KNN) classifier on the same dataset and compare your results. Finally, you will record a short screen-share with narration explaining your methods and observations.

You may work alone or with a partner. You may choose to work with the provided Penguins dataset, or select your own pre-cleaned dataset from the links below (I have suggested a few datasets as a guide, but you are welcome to select something different with approval).  The most important detail regarding your data-set is that your data needs to lend itself to classification.  For example, an iris with a sepal length of x and a petal width of y can be classified as ‘Setosa’. I also recommend that you use github codespaces, as you will need access to command-line tools that are unavailable in VS Code for EDU.

[UCI Machine Learning Repository](https://archive.ics.uci.edu/datasets)
 - Iris (classic 3-class classification)
 - Mushroom (binary classification: edible/poisonous)
 - Student Performance (predict grades, numeric features)

[Kaggle Datasets](https://www.kaggle.com/datasets)   *Note: For Kaggle, I will have to download the data for you and post on a shared drive.
 - Titanic survival dataset (binary classification)
 - Heart disease dataset (binary classification)
 - Breast cancer diagnosis (binary)
 - Penguins dataset (same as Kira, already cleaned)

---

**Team Members:** 
- Jax

**Dataset Used:**  
Fertility

**Source:**  
UCI

**Target Variable (What we are predicting):**  
Altered fertility

**Features Used:**  
- Accident or serious trauma
- Surgical intervention
- Smoking habits

**[Video Review](https://)**

## Human Algorithm
### Pseudo-Code
If no accident and no surgery and yes smoking:
   Flag
If no accident and yes surgery and not max smoking:
   Flag

When examining the data and visualizations, we focused on these features because they showed the clearest results with only binary data.
The plots showed that there was clustering with these features, so we considered values that met these clusters. 
From the summary tables and visualizations, it appeared that a combination of smoking, surgery, or accident could influence classification, which led us to choosing these specific points in our decision rules.

### Confusion Matrix

Accuracy: 61.11%

| Actual \ Predicted |  N |  O |
|--------------------|---------|
| N                  | 51 | 28 |
| O                  | 7  | 4  |
One example where our algorithm worked well is when the inputs were in most locations, leading to a correct prediction of N because most of the data was normal.

An example where the algorithm did not perform as expected is when the inputs were altered, resulting in a prediction of N instead of O, which may have happened because the data is very binary, meaning the clumps were not totally accurate.

These examples of success and failure highlight patterns in the data or limitations in our rules, such as the incredibly binary nature of it.

<img width="315" height="334" alt="image" src="https://github.com/user-attachments/assets/23ee1e49-da76-47c2-97b8-c8fbcbef179c" />

## Machine Learning Model

I test a bunch of different machine learning models to find the best one, since KNN wasn't working very well for each. There are graphs for each of them in the ml_model plot folder. I found the best result with NuSVC with a nu value of .2 and a split of 30% training. Even though the accuracy percentage is not very good, it has a much better accuracy when the actual is altered, which is best in medical applications. This model gave me no false negatives, meaning everyone who would be returned a negative result is definitely normal. While it has a greater error with predicted altered actual normal, it catches every altered case which I believe is the most important in this case. 

### Confusion Matrix

Accuracy: 55.71%

| Actual \ Predicted | Class 1 | Class 2 |
|-------------------|---------|---------|
| N                 |   31    |   31    |
| O                 |    0    |    8    |

The table/visualization shows a clear pattern where the model predicts ___ when ___, indicating a strong relationship between these features.

The confusion matrix reveals that the model most often confuses ___ with ___, suggesting these classes have similar feature values.

Compared to the human algorithm, the KNN model shows different behavior when ___, as seen in the ___ visualization.

<img width="315" height="334" alt="image" src="https://github.com/user-attachments/assets/199ae59d-3470-40c6-9669-60e62b211619" />
