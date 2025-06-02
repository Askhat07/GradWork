import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
import pickle
import warnings

warnings.filterwarnings('ignore')

# Loading and reading the data
dataFrame = pd.read_csv("C:/Users/ashat/PycharmProjects/GraduateWork/Crop_Recommendation/Crop_recommendation.csv")

# About the data
print(dataFrame.head(), '\n')

print(dataFrame.info(), '\n')

print(dataFrame.describe(), '\n')

print("Shape of the DataFrame: ", dataFrame.shape, '\n')

print(dataFrame.isna().sum(), '\n')

print(dataFrame['label'].unique(), '\n')

print(dataFrame['label'].value_counts(), '\n')

# Exploratory Data Analysis
sns.heatmap(dataFrame.isnull(), cmap="coolwarm")
plt.show()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.distplot(dataFrame['temperature'], color="orange", bins=15, hist_kws={'alpha':0.2})
plt.subplot(1, 2, 2)
sns.distplot(dataFrame['ph'], color="purple", bins=15, hist_kws={'alpha':0.2})
plt.show()

sns.countplot(y='label', data=dataFrame, palette="plasma_r")
plt.show()

sns.pairplot(dataFrame, hue='label')
plt.show()

sns.jointplot(x='rainfall', y='humidity', data=dataFrame[(dataFrame['temperature'] < 30) & (dataFrame['rainfall'] > 120)], hue="label")
plt.show()

sns.jointplot(x='K', y='N', data=dataFrame[(dataFrame['N'] > 40)], hue='label')
plt.show()

sns.jointplot(x='K', y='humidity', data=dataFrame, hue='label', size=8, s=30, alpha=0.7)
plt.show()

sns.boxplot(y='label', x='ph', data=dataFrame)
plt.show()

sns.boxplot(y='label', x='P', data=dataFrame[dataFrame['rainfall'] > 150])
plt.show()

sns.lineplot(data=dataFrame[(dataFrame['humidity'] < 65)], x='K', y='rainfall', hue='label')
plt.show()

# Seperating features
features = dataFrame[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
targets = dataFrame['label']
labels = dataFrame['label']

# Correlation visualization between features
sns.heatmap(features.corr(), annot=True)
plt.show()

# Splitting into train and test data
xTrain, xTest, yTrain, yTest = train_test_split(features, targets, test_size=0.2, random_state=2)
print(len(xTrain), len(yTrain), len(xTest), len(yTest), '\n')

# Classification using Random Forest
randomForest = RandomForestClassifier(n_estimators=20, random_state=0)
randomForest.fit(xTrain, yTrain)
predictValues = randomForest.predict(xTest)
acc = metrics.accuracy_score(yTest, predictValues)
print("Random Forrest's accuracy is: ", acc, '\n')
print(classification_report(yTest, predictValues), '\n')

# Cross validation score
score = cross_val_score(randomForest, features, targets, cv=5)
print(score, '\n')

# Saving trained Random Forest model
# Random_Forest_name = 'RandomForrest.pkl'
# Random_Forest_pkl = open(Random_Forest_name, 'wb')
# pickle.dump(randomForest, Random_Forest_pkl)
# Random_Forest_pkl.close()

# Making a prediction
testPredictData = np.array([[80, 50, 40, 25, 80, 6.5, 250]])
testPredict = randomForest.predict(testPredictData)
print(testPredict)