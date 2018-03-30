'''
   Develop Your First Neural Network in Python With Keras Step-By-Step
   https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/ 

   The steps you are going to cover in this tutorial are as follows:
        Load Data.
        Define Model.
        Compile Model.
        Fit Model.
        Evaluate Model.
        Tie It All Together.
'''

from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("data/pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# create model
'''
    Models in Keras are defined as a sequence of layers. 
    We create a Sequential model and add layers one at a time until we are happy with our network topology.
    We use a fully-connected network structure with three layers.
    Fully connected layers are defined using the Dense class
'''
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))   #input_dim=8 means 8 input variables
model.add(Dense(8, activation='relu'))  # 8 neurons in the laryer, 
model.add(Dense(1, activation='sigmoid'))
'''
initialize the network weights to a small random number generated from a uniform distribution (‘uniform‘), 
in this case between 0 and 0.05 because that is the default uniform weight initialization in Keras.
'''

'''
Compiling the model uses backend library such as Theano or TensorFlow. 
The backend automatically chooses the best way to represent the network for training
'''
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

'''
Fit the model
    nepochs is the number of iterations
    batch_size is the number of instances that are evaluated before a weight update in the network is performed
'''
model.fit(X, Y, epochs=150, batch_size=10)

'''
Evaluate Model

'''
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))