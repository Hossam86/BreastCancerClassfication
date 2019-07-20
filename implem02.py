import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split

plt.style.use("ggplot")
pd.set_option("display.max_columns", 500)

