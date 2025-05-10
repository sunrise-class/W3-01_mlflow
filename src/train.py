import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import argparse

def load_data():
    df = pd.read_csv("/Users/channary/Documents/Learn/sunrise/AI_course/homework/MLflow/titanic-mlflow/data/titanic.csv")
    df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']].dropna()
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=5)
args = parser.parse_args()

# Load Data
X_train, X_test, y_train, y_test = load_data()

with mlflow.start_run():
    clf = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(clf, "model")