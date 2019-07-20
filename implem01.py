import numpy as nb
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

bc=load_breast_cancer()
X=bc.data
y=bc.target

# creat our test/train split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)

## build our models
decision_tree = DecisionTreeClassifier()
random_forest = RandomForestClassifier(n_estimators=100)

## Train the classifiers
decision_tree.fit(X_train, y_train)
random_forest.fit(X_train, y_train)

# Create Predictions
dt_pred = decision_tree.predict(X_test)
rf_pred = random_forest.predict(X_test)

# Check the performance of each model
print('Decision Tree Model')
print(classification_report(y_test, dt_pred, target_names=bc.target_names))

print('Random Forest Model')
print(classification_report(y_test, rf_pred, target_names=bc.target_names))

#Graph our confusion matrix
dt_cm = confusion_matrix(y_test, dt_pred)
rf_cm = confusion_matrix(y_test, rf_pred)

# Get our features and weights
feature_list = sorted(zip(map(lambda x: round(x, 2), random_forest.feature_importances_), bc.target_names),
             reverse=True)

# Print them out
print('feature\t\timportance')
print("\n".join(['{}\t\t{}'.format(f,i) for i,f in feature_list]))
print('total_importance\t\t',  sum([i for i,f in feature_list]))