from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,classification_report

bc=load_breast_cancer()
X=bc.data
y=bc.target
