
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.utils.vis_utils import plot_model
import numpy
import pydot
import graphviz

def create_model():
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    # We use sigmoid to ensure that our network
    # output is between 0 and 1
    model.add(Dense(1, activation='sigmoid'))
    return model

# fix random seed for reproducibility
numpy.random.seed(7)

# Load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# Split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# Create model
model = create_model()
plot_model(model, to_file='model1.png', show_shapes=True)

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)

# Evaluate the model
scores = model.evaluate(X, Y)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))