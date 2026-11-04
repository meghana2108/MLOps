from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def load_data():
    iris = load_iris()
    return iris.data, iris.target

def split_data(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=12)