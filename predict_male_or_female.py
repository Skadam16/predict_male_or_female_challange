"""Classify_male_or_female
    Siraj Raval challenge
    Written by: Samruddhi Kadam"""


#Importing dependencies
from sklearn import tree, svm
from sklearn.linear_model.stochastic_gradient import SGDClassifier

#Input data
#[height, weight, shoe size] data
X = [[181,80,44],[177,70,43],[160,60,38],[154,54,37],[166,65,40],
		[190,90,47],[175,64,39],[177,70,40],[159,55,37],
		[171,75,42],[181,85,43]]

#Corresponding gender tags
Y = ['male','female','female','female','male','male',
		'male','female','male','female','male']

# Decision Tree classifier- takes in the input data to predict whether male or female
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(X,Y)

#Support Vector Machine Classifier
classifier1 = svm.SVC()
classifier1 = classifier1.fit(X,Y)

#Stochastic Gradient Descent 
clf = SGDClassifier()
clf = clf.fit(X,Y)

#Prediction step
#prediction for decision Trees
prediction = classifier.predict([[172,75,35]])
print(prediction)

#Prediction for Support Vector Machines
prediction1 =classifier1.predict([[177,70,43]])
print(prediction1)

#Prediction for Stochastic Gradient Descent
pred = clf.predict([[172,75,35]])
print(pred)