

from source import optimizers,costs
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import cPickle
import load_data
from adv_auto_mnist import AdversarialAutoencoderMNIST

import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import os
import errno

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise



def train(latent_dim = 2, #dimension of latent variable z
          z_prior = 'gaussian', # 'gaussian' or 'uniform'
          lamb = 10., #ratio between reconstruction and adversarial cost
          recon_obj_type = 'CE', #objective function on reconstruction ( 'CE'(cross ent.) or 'QE'(quadratic error) )
          initlal_learning_rate = 0.002,
          learning_rate_decay=1.0,
          num_epochs=50,
          batch_size=100,
          save_filename='trained_model',
          seed=1):


    numpy.random.seed(seed=seed)

    dataset = load_data.load_mnist_full()

    x_train,_ = dataset[0]
    x_test,_ = dataset[1]

    model = AdversarialAutoencoderMNIST(latent_dim=latent_dim,z_prior=z_prior)

    x = T.matrix()

    loss_for_training,_,adv_loss_for_training = costs.adversarial_autoenc_loss(x=x,
                                          enc_f=model.encode_train,
                                          dec_f=model.decode_train,
                                          disc_f=model.D_train,
                                          p_z_sampler=model.sample_from_prior,
                                          obj_type=recon_obj_type,
                                          lamb=numpy.asarray(lamb,dtype=theano.config.floatX))

    _,recon_loss,adv_loss = costs.adversarial_autoenc_loss(x=x,
                                          enc_f=model.encode_test,
                                          dec_f=model.decode_test,
                                          disc_f=model.D_test,
                                          p_z_sampler=model.sample_from_prior,
                                          obj_type=recon_obj_type,
                                          lamb=numpy.asarray(lamb,dtype=theano.config.floatX))

    optimizer_recon = optimizers.ADAM(cost=loss_for_training,
                                      params=model.model_params,
                                      alpha=numpy.asarray(initlal_learning_rate,dtype=theano.config.floatX))
    optimizer_adv = optimizers.ADAM(cost=adv_loss_for_training,
                                    params=model.D_params,
                                    alpha=numpy.asarray(initlal_learning_rate,dtype=theano.config.floatX))

    index = T.iscalar()

    f_training_model = theano.function(inputs=[index], outputs=loss_for_training, updates=optimizer_recon.updates,
                              givens={
                                  x:x_train[batch_size*index:batch_size*(index+1)]})
    f_training_discriminator = theano.function(inputs=[index], outputs=adv_loss_for_training, updates=optimizer_adv.updates,
                              givens={
                                  x:x_train[batch_size*index:batch_size*(index+1)]})


    f_recon_train = theano.function(inputs=[index], outputs=recon_loss,
                              givens={
                                  x:x_train[batch_size*index:batch_size*(index+1)]})
    f_adv_train = theano.function(inputs=[index], outputs=adv_loss,
                              givens={
                                  x:x_train[batch_size*index:batch_size*(index+1)]})
    f_recon_test = theano.function(inputs=[index], outputs=recon_loss,
                              givens={
                                  x:x_test[batch_size*index:batch_size*(index+1)]})
    f_adv_test = theano.function(inputs=[index], outputs=adv_loss,
                              givens={
                                  x:x_test[batch_size*index:batch_size*(index+1)]})

    f_lr_decay_recon = theano.function(inputs=[],outputs=optimizer_recon.alpha,
                                 updates={optimizer_recon.alpha:theano.shared(numpy.array(learning_rate_decay).astype(theano.config.floatX))*optimizer_recon.alpha})
    f_lr_decay_adv = theano.function(inputs=[],outputs=optimizer_adv.alpha,
                                 updates={optimizer_adv.alpha:theano.shared(numpy.array(learning_rate_decay).astype(theano.config.floatX))*optimizer_adv.alpha})

    randix = RandomStreams(seed=numpy.random.randint(1234)).permutation(n=x_train.shape[0])
    f_permute_train_set = theano.function(inputs=[],outputs=x_train,updates={x_train:x_train[randix]})

    statuses = {}
    statuses['recon_train'] = []
    statuses['adv_train'] = []
    statuses['recon_test'] = []
    statuses['adv_test'] = []

    n_train = x_train.get_value().shape[0]
    n_test = x_test.get_value().shape[0]

    sum_recon_train = numpy.sum(numpy.array([f_recon_train(i) for i in xrange(n_train/batch_size)]))*batch_size
    sum_adv_train = numpy.sum(numpy.array([f_adv_train(i) for i in xrange(n_train/batch_size)]))*batch_size
    sum_recon_test = numpy.sum(numpy.array([f_recon_test(i) for i in xrange(n_test/batch_size)]))*batch_size
    sum_adv_test = numpy.sum(numpy.array([f_adv_test(i) for i in xrange(n_test/batch_size)]))*batch_size
    statuses['recon_train'].append(sum_recon_train/n_train)
    statuses['adv_train'].append(sum_adv_train/n_train)
    statuses['recon_test'].append(sum_recon_test/n_test)
    statuses['adv_test'].append(sum_adv_test/n_test)
    print "[Epoch]",str(-1)
    print  "recon_train : " , statuses['recon_train'][-1], "adv_train : ", statuses['adv_train'][-1], \
            "recon_test : " , statuses['recon_test'][-1],  "adv_test : ", statuses['adv_test'][-1]

    z = model.encode_test(input=x)
    f_enc = theano.function(inputs=[],outputs=z,givens={x:dataset[1][0]})
    def plot_latent_variable(epoch):
        output = f_enc()
        plt.figure(figsize=(8,8))
        color=cm.rainbow(numpy.linspace(0,1,10))
        for l,c in zip(range(10),color):
            ix = numpy.where(dataset[1][1].get_value()==l)[0]
            plt.scatter(output[ix,0],output[ix,1],c=c,label=l,s=8,linewidth=0)
        plt.xlim([-5.0,5.0])
        plt.ylim([-5.0,5.0])
        plt.legend(fontsize=15)
        plt.savefig('z_epoch' + str(epoch) + '.pdf')

    print "training..."
    make_sure_path_exists("./trained_model")

    for epoch in xrange(num_epochs):
        cPickle.dump((model,statuses),open('./trained_model/'+'tmp-' + save_filename,'wb'),cPickle.HIGHEST_PROTOCOL)
        f_permute_train_set()
        ### update parameters ###
        for i in xrange(n_train/batch_size):
            ### Optimize model and discriminator alternately ###
            f_training_discriminator(i)
            f_training_model(i)
        #########################

        if(latent_dim == 2):
            plot_latent_variable(epoch=epoch)

        sum_recon_train = numpy.sum(numpy.array([f_recon_train(i) for i in xrange(n_train/batch_size)]))*batch_size
        sum_adv_train = numpy.sum(numpy.array([f_adv_train(i) for i in xrange(n_train/batch_size)]))*batch_size
        sum_recon_test = numpy.sum(numpy.array([f_recon_test(i) for i in xrange(n_test/batch_size)]))*batch_size
        sum_adv_test = numpy.sum(numpy.array([f_adv_test(i) for i in xrange(n_test/batch_size)]))*batch_size
        statuses['recon_train'].append(sum_recon_train/n_train)
        statuses['adv_train'].append(sum_adv_train/n_train)
        statuses['recon_test'].append(sum_recon_test/n_test)
        statuses['adv_test'].append(sum_adv_test/n_test)
        print "[Epoch]",str(epoch)
        print  "recon_train : " , statuses['recon_train'][-1], "adv_train : ", statuses['adv_train'][-1], \
                "recon_test : " , statuses['recon_test'][-1],  "adv_test : ", statuses['adv_test'][-1]

        f_lr_decay_recon()
        f_lr_decay_adv()

    make_sure_path_exists("./trained_model")
    cPickle.dump((model,statuses),open('./trained_model/'+save_filename,'wb'),cPickle.HIGHEST_PROTOCOL)
    return model,statuses

if __name__=='__main__':
    train()
