import theano.tensor as T
from layer import Layer

class Sigmoid(Layer):

    def forward(self,x):
        return T.nnet.sigmoid(x)

def sigmoid(x):
    return Sigmoid()(x)