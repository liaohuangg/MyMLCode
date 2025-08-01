import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor, XGBRegressor
from sklearn.datasets import load_digits
from sklearn.datasets import load_diabetes
# import sklearn
print("XGBoost version:", xgb.__version__)