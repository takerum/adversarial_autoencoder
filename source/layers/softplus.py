import theano.tensor as T
from layer import Layer

class Softplus(Layer):

    def forward(self,x):
        return T.nnet.softplus(x)

def softplus(x):
    return Softplus()(x)