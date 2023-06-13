# Importing the relevant classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingCVClassifier
from sklearn.tree import DecisionTreeClassifier

# Importing the relevant functions to obtain performance metrics
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time

# Import relevant libraries/ functions for visualizations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# DATA PREPARATION
# Importing the data
# --http://archive.ics.uci.edu/ml/datasets/banknote+authentication#

# Defining the attribute names
columnlabels = ["Variance", "Skewness", "Kurtosis", "Entropy", "Class"]

# Read the .CSV file that was downloaded
# from the UCI Data Repository and assign the respective column labels
data = pd.read_csv('bn_authentication.csv', delimiter=',', names=columnlabels)

# Confirm all attributes were imported correctly
print(data.info())

# DATA UNDERSTANDING

# Create pie chart to view distribution of Target Variable
class_label = ['Counterfeit', 'Genuine']
data['Class'].value_counts().plot.pie(autopct='%1.1f%%', labels=class_label)
plt.title('Distribution of The Target Variable')

# Histograms of Each Attribute
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
axes = axes.flatten()

# Loop through each attribute and plot on subplots
for i, attribute in enumerate(columnlabels[0:4]):
    sns.histplot(x=attribute, data=data, kde=True, hue='Class', ax=axes[i])
    axes[i].legend(title='Class', labels=['Genuine', 'Counterfeit'])
    axes[i].set_title(f'Histogram for {attribute}')

# Adjust spacing between subplots
plt.tight_layout()
plt.show()

# ALGORITHM PERFORMANCE

# Note: The results will be different from the report. To keep
# the same result every time you run the code,we can add
# 'randomstate' parameters to the relevant areas

# Dividing the algorithm into the target variable and the attributes
X = data.values[:, 0:3]
Y = data.values[:, 4]

# Assigning the ML Algorithms to variable names with default settings
kneigh = KNeighborsClassifier()
rndfor = RandomForestClassifier()
dctree = DecisionTreeClassifier()
gaus = GaussianNB()
lr = LogisticRegression()

# Creating the Stacked Classifier with KNN and Random Forest
sclf = StackingCVClassifier(classifiers=[kneigh, rndfor],
                            use_probas=True,
                            meta_classifier=lr)

# Running the 5-Fold Cross Validation for the 5 ML Algorithms
scores = {}
mean_scores = []
runtimes = []

for clf, label in zip([kneigh, rndfor, gaus, dctree, sclf],
                      ['KNN',
                       'Random Forest',
                       'Naive Bayes',
                       'Decision Tree Classifier',
                       'StackingClassifier']):
    start_time = time.time()  # Start timing before running the Cross Validation
    scores[clf] = cross_val_score(clf, X, Y, cv=5, scoring='accuracy')
    end_time = time.time()  # End timing after running the Cross Validation

    runtime = end_time - start_time
    mean_score = scores[clf].mean() * 100
    mean_scores.append((label, mean_score, runtime))

# Print mean scores
for label, mean_score, runtime in mean_scores:
    print(f'{label} mean score: {mean_score}, runtime: {runtime} seconds')

print('\n')

# Create the dataset to store the results in a manageable format
df_mean_scores = pd.DataFrame(mean_scores, columns=['Classifier', 'Mean Accuracy', 'Runtime'])

# Plotting the bar chart using seaborn
fig, ax1 = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
bar_avg = sns.barplot(data=df_mean_scores, x='Classifier', y='Mean Accuracy', ax=ax1)
ax1.set_xlabel('Classifier')
ax1.set_ylabel('Mean Accuracy')
ax1.set_title('Mean Accuracy and Runtimes of Each Classifier')

# Add values above the bars as percentages
for i, bar in enumerate(bar_avg.patches):
    value = mean_scores[i][1]
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), '{:.2f}%'.format(value), ha='center', va='bottom')

# Create a secondary y-axis for runtimes
ax2 = ax1.twinx()
ax2.set_ylabel('Runtime (seconds)')

# Plotting the runtimes as a line chart on the secondary y-axis
line_runtime = sns.lineplot(data=df_mean_scores, x='Classifier', y='Runtime', marker='o', ax=ax2, color='black')
line_runtime.set(ylim=(0, max(df_mean_scores['Runtime']) * 1.2))

plt.tight_layout()
plt.show()

# Split the data into Training and Testing sections for the confusion matrix
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Training the KNN algorithm and Testing it to obtain its predictions
kneigh.fit(x_train, y_train)
y_pred = kneigh.predict(x_test)

# Creating the Confusion Matrix for the predictions made above
cm = confusion_matrix(y_test, y_pred, labels=kneigh.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=kneigh.classes_)
disp.plot()
plt.show()

# Define the hyperparameter tuning grid
param_grid = {
    'n_neighbors': [3, 5, 7],  # try different values for n_neighbors
    'weights': ['uniform', 'distance'],
    'algorithm': ['brute', 'kd_tree', 'ball_tree'],  # try different algorithms
    'leaf_size': [20, 30, 40],  # try different leaf sizes
    'p': [1, 2]  # try different distance metrics (Manhattan and Euclidean)
}

# Grid search to find the parameters that give the most accurate result.
grid_search = GridSearchCV(kneigh, param_grid, cv=5)
grid_search.fit(x_train, y_train)

best_params = grid_search.best_params_

# assigning the parameters to a new KNN algorithm
kneigh_best = KNeighborsClassifier(**best_params)

print(best_params, '\n')

# Performing 5-Fold Cross Validation for the Default and Best Parameters
mean_scores_param = []

for clf, parameter in zip([kneigh, kneigh_best],
                          ['Default Parameters',
                           'Parameters for Best Accuracy']):
    start_time = time.time()  # Start timing
    scores[clf] = cross_val_score(clf, X, Y, cv=5, scoring='accuracy')
    end_time = time.time()  # End timing

    runtime = end_time - start_time
    mean_score = scores[clf].mean() * 100
    mean_scores_param.append((parameter, '{:.2f}%'.format(mean_score), '{:.3f}'.format(runtime)))

# Print mean scores of both parameter settings
for parameter, mean_score, runtime in mean_scores_param:
    print(f'{parameter} Cross Validation Mean Score: {mean_score}, Cross Validation Runtime: {runtime} seconds')
