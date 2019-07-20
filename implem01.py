import numpy as nb
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

bc=load_breast_cancer()
X=bc.data
y=bc.target

# creat our test/train split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)
