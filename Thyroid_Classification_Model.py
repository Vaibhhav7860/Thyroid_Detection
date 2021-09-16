# Importing necessary Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math

# Importing train-test-split for creating validation (testing) set.

from sklearn.model_selection import train_test_split

# Importing Decision Tree Classifier.

from sklearn.tree import DecisionTreeClassifier

from sklearn import tree

#  Importing feature selection and chi2  for selecting the best features

from sklearn.feature_selection import SelectKBest, chi2

# Importing Random for shuffling the testing data

import random

# Importing confusion matrix for calculating the confusion matrix
from sklearn.metrics import confusion_matrix

# Loading our dataset.

missing_value = ["N/a", "na", np.nan]
df = pd.read_csv("Thyroid_Dataset.csv", na_values=missing_value)

print(df)


# Displaying first 5 rows of our dataset
print(df.head())

# Displaying the size of our dataset.
print("Dataset Size : ",df.shape)

# Performing Data Cleaning

print(df.isnull().any())    # Checks whether any attribute has a null value or not.
                            # If it detects a null value under any attribute then it returns false for that attribute otherwise true.

print(df.isnull().sum())    # It returns the total count of null values under each attribute.

# Since all the parameters correspond to 0 and False therefore there are no null values in our dataset



# Seprating Independent and Dependent Attributes

y = df['Class']                             # Class attribute is our target (dependent) variable
x = df.drop(['Class','Name','Type'], axis= 1)             # Remaining attributes are independent variables

# Applying SelectKBest class to extract top 6 best features

bestFeatures = SelectKBest(score_func=chi2, k=6)
fit = bestFeatures.fit(x,y)

dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)

# Concatenating both the dataframe for better visualizations

featureScores = pd.concat([dfcolumns, dfscores], axis = 1)
featureScores.columns = ['Attribute','Score']     # Naming the dataframe columns

print(featureScores)


# Printing the 6 best features

print(featureScores.nlargest(6,'Score'))

# Plotting a bar graph of feature importance for better visualization

plt.figure(figsize=(12,6))

f = featureScores.nlargest(6,'Score')
indexes_top6 = list(f['Attribute'])
print(indexes_top6)
scores_top6 = list(f['Score'])
print(scores_top6)
plt.bar(indexes_top6,scores_top6,width=0.6)
plt.xlabel("Top 6 Attributes ")
plt.ylabel("Scores ")
plt.title(" Graphical Representation of Top 6 Features ")
plt.show()



# Taking top 6 attributes of our dataset and now we will use these attributes to train our model.
x_new = df[['TSH', 'T4', 'Age', 'T3', 'Sex', 'Married']]

# Creating the training and testing set

X_train, X_test, Y_train, Y_test = train_test_split(x_new, y, random_state=101, stratify=y, test_size=0.25)

# Checking Distribution in Training Set
print(Y_train.value_counts(normalize=True))

# Checking Distribution in Validation (Testing) Set
print(Y_test.value_counts(normalize= True))

# Shape of training set
print(X_train.shape, Y_train.shape)        # 975 observations in training set

# Shape of testing set
print(X_test.shape, Y_test.shape)          # 325 observations in testing set



# Changing the max depth of the tree

train_accuracy = []
test_accuracy = []
for depth in range(1, 11):
    dt_model = DecisionTreeClassifier(max_depth=depth, random_state= 10)
    dt_model.fit(X_train, Y_train)
    train_accuracy.append(dt_model.score(X_train, Y_train))
    test_accuracy.append(dt_model.score(X_test, Y_test))

frame = pd.DataFrame({'Max_Depth':range(1,11), 'Training_Accuracy':train_accuracy, 'Testing_Accuracy':test_accuracy})
print(frame)



# Predictions on testing set

predictions = dt_model.predict(X_test)

print(predictions)

# Plotting a graph for training accuracy and testing accuracy of our model.

plt.figure(figsize= (12, 6))
plt.plot(frame['Max_Depth'], frame['Training_Accuracy'], marker='o')
plt.plot(frame['Max_Depth'], frame['Testing_Accuracy'], marker='o')
plt.xlabel('Depth of tree')
plt.ylabel('Performance')
plt.legend(['Training', 'Testing'])
plt.title('Training Accuracy vs Testing Accuracy')
plt.show()


LABEL = ['No Thyroid', 'Hypo Thyroid', 'Hyper Thyroid']

predictions_copy = predictions

random.shuffle(predictions_copy)
a = LABEL[predictions_copy[0]-1]

z = df['Name']
print(z)

print(predictions_copy[0])



print("Prediction is : ", a)

if a == "Hypo Thyroid" or a == "Hyper Thyroid":
    print("Yes ! Treatment is required")
else:
    print("No ! There is no need of treatment ")


# Calculating total number of people who require treatment
count_ones = list(predictions_copy).count(1)
count_twos = list(predictions_copy).count(2)
count_three = list(predictions_copy).count(3)

N = count_twos + count_three    # N represents total no. of patients who need treatment.

# There are total 318 patients out of 325 who need treatment.

Entropy = 0
for i in range(1,N+1):
    Entropy += predictions[i-1]*math.log(predictions[i-1],2)
Entropy = (-1)*Entropy

print("Entropy is : ", Entropy)

EffectOfTreatment = Entropy*(N - count_ones)
print("Effect of Treatment is : ", abs(EffectOfTreatment))


# Computing the confusion matrix of the model.
confusionMatrix = confusion_matrix(y_true=Y_test,y_pred=predictions)
print(confusionMatrix)

# Predicted Values :
print("Predicted : ")
predicted = []
for i in range(len(predictions)):
    predicted.append(predictions[i])

print(predicted)

# Actual Values :
print("Actual : ")

actual = []
for i in range(len(Y_test)):
    actual.append(list(Y_test)[i])

print(actual)

# Computing TruePositive, FalsePositive, TrueNegative, FalseNegative Values

class_id = set(actual).union(set(predicted))
TruePositive = 0
FalsePositive = 0
TrueNegative = 0
FalseNegative = 0

for index ,_id in enumerate(class_id):
    for i in range(len(predicted)):
        if actual[i] == predicted[i] == _id:
            TruePositive += 1
        if predicted[i] == _id and actual[i] != predicted[i]:
            FalsePositive += 1
        if actual[i] == predicted[i] != _id:
            TrueNegative += 1
        if predicted[i] != _id and actual[i] != predicted[i]:
            FalseNegative += 1

print(class_id)
print("True Positive Value is : ", TruePositive)
print("False Positive Value is : ", FalsePositive)
print("True Negative Value is : ",TrueNegative)
print("False Negative Value is : ",FalseNegative)

# Computing Accuracy
print("Accuracy : ")

Correct_Assesments = TruePositive + TrueNegative
Total_Assesments = TruePositive + TrueNegative + FalsePositive + FalseNegative

Accuracy = Correct_Assesments/Total_Assesments

print(Accuracy)

# Computing Precision
print("Precision : ")

Precision = TruePositive/(TruePositive+FalsePositive)
print(Precision)

# Computing Recall
print("Recall (Sensitivity) : ")
Recall = TruePositive/(TruePositive+FalseNegative)
print(Recall)

# Computing Specificity
print("Specificity : ")
Specificity = TrueNegative/(TrueNegative + FalsePositive)
print(Specificity)

# Computing F1 Score
print("F1 Score : ")
F1_Score = 2*(Precision*Recall)/(Precision + Recall)
print(F1_Score)
