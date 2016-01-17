import theano.tensor as T

def binary_cross_entropy_loss(x,t,forward_func):
    print "costs/binary_cross_entropy_loss"
    y = forward_func(x)
    return _binary_cross_entropy_loss(y,t)

def _binary_cross_entropy_loss(y,t):
    return -T.mean(T.sum(t*T.log(y) + (1-t)*T.log(1-y),axis=1))
