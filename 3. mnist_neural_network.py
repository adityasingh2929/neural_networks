
# Importing the necessary libraries:

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.datasets import mnist


# Loading the mnist dataset

(x_train,y_train), (x_test,y_test) = mnist.load_data()


# Normalizing the data, since img pixel values go from 0 to 255, we normalized them from 0 to 1

x_train = x_train/255.0
x_test = x_test/255.0


# Converting labels into categorical format by performing One-Hot encoding.... initially there was one feature named label...which held values from 0-9 but now we've made 10 feature columns, from 0 to 9 and then either put 0 (inactive) or 1 (active) in those columns.

y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)


# Defining the structure of our neural network:
    # Input layer: tells the model the shape of each image, i.e 28 by 28 pixels
    # Flatten layer: converts the 2d img into a long 1d list of numbers so it can be processed by layers
    # 1st layer: contains a 1d array of img's info (784 values here) formed using input and flatten.
    # dense layer: this is where learning occurs  [128 neurons]  (the hidden layer)
    # dense output layer: gives probability for each digit (0-9), highest one becomes the prediction [10 neurons]

    # dense layer uses relu as its activation function which introduces non-linearity and hence helps model learn complex patterns.   =  ReLU(x)=max(0,x)

    # dense output layer uses softmax as its activation function as it helps converting raw numbers of the 10 neurons into probabilities...i.e it forces neurons to compete with each other and hence it ensures that all outputs are btw 0 and 1 and the total sum of all outputs are 1...hence making it easy for us to select what output has the highest probability.
    # Softmax(zi) = e^zi / summation (j=1 to k) e^zj    where k=10 digits in our case, basically its a prob. formula.


model = Sequential([
    Input(shape=(28,28)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])


# setting up for training the model, here we the model -> HOW TO LEARN  
    # Optimizer: It controls how the model adjust itself when it makes mistakes.
    # Loss: Measures how wrong the model's predictions are
    # Accuracy metric: Tracks how often the model predicts the correct digit

model.compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = ['accuracy']
)


# Here we're now training the model on the mnist dataset. The mnist dataset has 60,000 images..we've made batches of images of size 32.......so in total there's 60k/32 = 1875 batches....and we'll go over these 1875 batches for 5 times. 

model.fit(x_train,y_train, epochs=5, batch_size=32)


# We evaluate the model now on the basis of test dataset and find accuracy.

test_loss , test_acc = model.evaluate(x_test,y_test)
print(f"Test accuracy: {test_acc*100}%")