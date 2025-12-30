from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# Generating a synthetic dataset
X,y = make_classification(n_samples=1000,n_features=10, n_classes=2, random_state=42)

# get train and test from the generated dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)


# Initialize the perceptron
clf = Perceptron(
    max_iter=1000, # max number of epochs
    eta0=0.1,    # learning rate
    random_state=42,  # for reproducibility
    tol=1e-3,        # stop early if improvement is smaller than this
    shuffle=True   # shuffle data each epoch
)   
clf.fit(X_train,y_train)




accuracy = clf.score(X_test,y_test)
print(f"The accuracy is:  {accuracy*100}%")
