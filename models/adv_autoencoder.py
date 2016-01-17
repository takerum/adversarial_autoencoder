
class AdversarialAutoencoder(object):

    def __init__(self):
        self.model_params=None
        self.D_params=None
        raise NotImplementedError()

    ##### define encoder function #####
    def encode_train(self,input):
        return self.encode(input=input,train=True)
    def encode_test(self,input):
        return self.encode(input=input,train=False)

    def encode(self,input,train=True):
        raise NotImplementedError()

    ##### define decoder function #####
    def decode_train(self,input):
        return self.decode(input=input,train=True)
    def decode_test(self,input):
        return self.decode(input=input,train=False)

    def decode(self,input,train=True):
        raise NotImplementedError()

    ##### define discriminator function #####
    def D_train(self,input):
        return self.D(input=input,train=True)
    def D_test(self,input):
        return self.D(input=input,train=False)

    def D(self,input,train=True):
        raise NotImplementedError()
