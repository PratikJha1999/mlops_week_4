import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def test_model_accuracy():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42)
    model = joblib.load('iris_model.joblib')
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    assert acc > 0.8  # Example threshold