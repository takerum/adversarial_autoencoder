from models.adv_autoencoder import AdversarialAutoencoder
import source.layers as L
from theano.tensor.shared_randomstreams import RandomStreams
import numpy,theano
import theano.tensor as T

def get_normalized_vector(v):
    v = v / (1e-20 + T.max(T.abs_(v), axis=1, keepdims=True))
    v_2 = T.sum(v**2,axis=1,keepdims=True)
    return v / T.sqrt(1e-6+v_2)


class AdversarialAutoencoderMNIST(AdversarialAutoencoder):

    def __init__(self,latent_dim=2,z_prior='gaussian'):

        self.z_prior = z_prior

        self.enc_l1 = L.Linear((784,1000))
        self.enc_b1 = L.BatchNormalization(1000)
        self.enc_l2 = L.Linear((1000,1000))
        self.enc_b2 = L.BatchNormalization(1000)
        self.enc_l3 = L.Linear((1000,latent_dim))
        self.enc_b3 = L.BatchNormalization(latent_dim)

        self.dec_l1 = L.Linear((latent_dim,1000))
        self.dec_b1 = L.BatchNormalization(1000)
        self.dec_l2 = L.Linear((1000,1000))
        self.dec_b2 = L.BatchNormalization(1000)
        self.dec_l3 = L.Linear((1000,784))

        self.D_l1 = L.Linear((latent_dim,500))
        self.D_b1 = L.BatchNormalization(500)
        self.D_l2 = L.Linear((500,500))
        self.D_b2 = L.BatchNormalization(500)
        self.D_l3 = L.Linear((500,1))

        self.model_params = self.enc_l1.params + self.enc_l2.params + self.enc_l3.params \
                          + self.dec_l1.params + self.dec_l2.params + self.dec_l3.params \
                          + self.enc_b1.params + self.enc_b2.params + self.enc_b3.params \
                          + self.dec_b1.params + self.dec_b2.params
        self.D_params = self.D_l1.params + self.D_l2.params + self.D_l3.params
        self.rng = RandomStreams(seed=numpy.random.randint(1234))


    def encode(self,input,train=True):
        h = input
        h = self.enc_l1(h)
        h = self.enc_b1(h,train=train)
        h = L.relu(h)
        h = self.enc_l2(h)
        h = self.enc_b2(h,train=train)
        h = L.relu(h)
        h = self.enc_l3(h)
        h = self.enc_b3(h,train=train)
        return h

    def decode(self,input,train=True):
        h = input
        h = self.dec_l1(h)
        h = self.dec_b1(h,train=train)
        h = L.relu(h)
        h = self.dec_l2(h)
        h = self.dec_b2(h,train=train)
        h = L.relu(h)
        h = self.dec_l3(h)
        h = L.sigmoid(h)
        return h

    def D(self,input,train=True):
        h = input
        h = self.D_l1(h)
        h = L.relu(h)
        h = self.D_l2(h)
        h = L.relu(h)
        h = self.D_l3(h)
        h = L.sigmoid(h)
        return h

    def sample_from_prior(self,z):

        ###### gausssian #######
        if(self.z_prior is 'gaussian'):
            return 1.0*self.rng.normal(size=z.shape,dtype=theano.config.floatX)

        ###### uniform ########
        elif(self.z_prior is 'uniform'):
            v = get_normalized_vector(self.rng.normal(size=z.shape,dtype=theano.config.floatX))
            r = T.power(self.rng.uniform(size=z.sum(axis=1,keepdims=True).shape,low=0,high=1.0,dtype=theano.config.floatX),1./z.shape[1])
            r = T.patternbroadcast(r,[False,True])
            return 2.0*r*v

        else:
            raise NotImplementedError()

