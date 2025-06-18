from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def load_mnist(train_size=0.857):  # 60,000 / 70,000 = ~0.857
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    x, y = mnist['data'], mnist['target'].astype(int)

    x = x.astype('float32') / 255
    x = x.reshape(-1, 784, 1)

    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    y = encoder.fit_transform(y.reshape(-1, 1))
    y = y.reshape(-1, 10, 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=42)

    return (x_train, y_train), (x_test, y_test)