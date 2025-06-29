from sklearn.datasets import load_iris

def test_iris_data_shape():
    iris = load_iris()
    assert iris.data.shape == (150, 4)
    assert len(iris.target) == 150

def test_iris_no_missing_values():
    iris = load_iris()
    assert not (iris.data != iris.data).any()  # Checks for NaNs