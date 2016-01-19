import theano
import theano.tensor as T
import numpy
from binary_cross_entropy_loss import _binary_cross_entropy_loss
from quadratic_loss import _quadratic_loss

def adversarial_autoenc_loss(x,enc_f,dec_f,disc_f,p_z_sampler,
                             obj_type,
                             lamb=numpy.asarray(1.0,dtype=theano.config.floatX)):

    z_q = enc_f(x)
    z_p = p_z_sampler(z_q)

    adv_loss = adversarial_loss(z_q=z_q , z_p=z_p ,disc_f=disc_f)
    recon_loss = reconstruction_loss(x=x ,z=z_q ,dec_f=dec_f ,obj_type=obj_type)

    return recon_loss - lamb*adv_loss, recon_loss, adv_loss


def adversarial_loss(z_p,z_q,disc_f):
    y_q = disc_f(z_q)
    y_p = disc_f(z_p)
    return -T.mean(T.log(y_p) + (T.log(1-y_q)))

def reconstruction_loss(x,z,dec_f,obj_type='QE'):
    x_ = dec_f(z)
    if obj_type == 'QE':
        return _quadratic_loss(x_,x)
    elif obj_type == 'CE':
        return _binary_cross_entropy_loss(x_,x)

