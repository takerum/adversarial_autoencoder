import theano.tensor as T

def cross_entropy_loss(x,t,forward_func):
    print "costs/cross_entropy_loss"
    y = forward_func(x)
    return _cross_entropy_loss(y,t)

def _cross_entropy_loss(y,t):
    if(t.ndim==1):
        return -T.mean(T.log(y)[T.arange(t.shape[0]), t])
    elif(t.ndim==2):
        return -T.mean(T.sum(t*T.log(y),axis=1))
