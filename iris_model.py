from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.DataFrame(data.target, columns=['Target'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)

# Preprocess the data (Standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Flatten the target variable
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Train the models
def training_model_dt():
    model = DecisionTreeClassifier()
    trained_model = model.fit(X_train, y_train)
    return trained_model

def training_model_rf():
    model = RandomForestClassifier()
    trained_model = model.fit(X_train, y_train)
    return trained_model


def training_model_lr():
    model = LogisticRegression()
    trained_model = model.fit(X_train, y_train)
    return trained_model

dt_model = training_model_dt()
rf_model = training_model_rf()
lr_model = training_model_lr()