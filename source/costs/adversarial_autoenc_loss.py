import theano
import theano.tensor as T
import numpy
from binary_cross_entropy_loss import _binary_cross_entropy_loss
from quadratic_loss import _quadratic_loss

def adversarial_autoenc_loss(x,enc_f,dec_f,disc_f,q_sampler,
                             obj_type,
                             lamb=numpy.asarray(1.0,dtype=theano.config.floatX)):

    z_p = enc_f(x)
    z_q = q_sampler(z_p)

    adv_loss = adversarial_loss(z_p=z_p ,z_q=z_q ,disc_f=disc_f)
    recon_loss = reconstruction_loss(x=x ,z=z_p ,dec_f=dec_f ,obj_type=obj_type)

    return recon_loss - lamb*adv_loss, recon_loss, adv_loss


def adversarial_loss(z_q,z_p,disc_f):
    y_p = disc_f(z_p)
    y_q = disc_f(z_q)
    return -T.mean(T.log(y_q) + (T.log(1-y_p)))

def reconstruction_loss(x,z,dec_f,obj_type='QE'):
    x_ = dec_f(z)
    if obj_type is 'QE':
        return _quadratic_loss(x_,x)
    elif obj_type is 'CE':
        return _binary_cross_entropy_loss(x_,x)

