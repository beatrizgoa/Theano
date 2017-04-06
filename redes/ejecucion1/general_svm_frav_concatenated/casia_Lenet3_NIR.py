# En este archivo de python se va a utilizar la base de datos de FRAV y Casia de imagenes  y se va a seguir la configuracion de la CNN del paper 'Learn convolutional neural network for face anti-spoofing'
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy
import timeit
from pylab import *
from logistic_sgd import LogisticRegression
from sklearn.svm import SVC
from layers import *
import sys
import theano
import theano.tensor
import auxiliar_functions
from sklearn.cross_validation import cross_val_score
import pickle
import os, argparse

#nkerns=[96, 256, 386, 384, 256]


def evaluate_lenet5_NIR(learning_rate, n_epochs, nkerns, batch_size, data):

    rng = numpy.random.RandomState(123456)

    train_set_x_rgb, test_set_x_rgb, valid_set_x_rgb, y_train, y_test, y_val = data

    print (numpy.array(train_set_x_rgb).shape, numpy.array(test_set_x_rgb).shape, numpy.array(valid_set_x_rgb).shape)
    print (numpy.array(y_train).shape, numpy.array(y_test).shape, numpy.array(y_val).shape)

    print((train_set_x_rgb[0]).shape)

    train_set_x_rgb = theano.shared(numpy.array(train_set_x_rgb, dtype= 'float32'), borrow=True)
    test_set_x_rgb = theano.shared(numpy.array(test_set_x_rgb, dtype= 'float32'), borrow=True)
    valid_set_x_rgb = theano.shared(numpy.array(valid_set_x_rgb, dtype= 'float32'), borrow=True)

    train_set_y = theano.shared(numpy.array(y_train, dtype='int32'), borrow=True)
    test_set_y = theano.shared(numpy.array(y_test, dtype='int32'), borrow=True)
    valid_set_y = theano.shared(numpy.array(y_val,dtype='int32'), borrow=True)


   # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x_rgb.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x_rgb.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x_rgb.get_value(borrow=True).shape[0]

    print("n_train_samples: %d" % n_train_batches)
    print("n_valid_samples: %d" % n_valid_batches)
    print("n_test_samples: %d" % n_test_batches)
    print("n_batches:")

    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    print("n_train_batches: %d" % n_train_batches)
    print("n_valid_batches: %d" % n_valid_batches)
    print("n_test_batches: %d" % n_test_batches)

    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')

    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
    is_train = T.iscalar('is_train')  # To differenciate between train and test

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print ('... building the model')


    ## RGB MODEL ##


    layer0_input = x.reshape((batch_size, 1, 128, 128))

    layer0 = LeNetConvPoolLRNLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 128, 128),
        filter_shape=(nkerns[0], 1, 11, 11),
        stride=(1, 1),
        lrn=True,
        poolsize=(2, 2),
        activation=theano.tensor.nnet.relu
    )

    layer1 = LeNetConvPoolLRNLayer(
        rng,
        input=layer0.output,
        # image_shape=(batch_size, nkerns[0], 27, 27),
        image_shape=(batch_size, nkerns[0], 59, 59),
        filter_shape=(nkerns[1], nkerns[0], 4, 4),
        lrn=True,
        poolsize=(2, 2),
        activation=theano.tensor.nnet.relu
    )

    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], 28, 28),
        filter_shape=(nkerns[2], nkerns[1], 3, 3),
        poolsize=(1, 1),
        activation=theano.tensor.nnet.relu

    )

    layer3 = LeNetConvPoolLayer(
        rng,
        input=layer2.output,
        image_shape=(batch_size, nkerns[2], 26, 26),
        filter_shape=(nkerns[3], nkerns[2], 3, 3),
        poolsize=(1, 1),
        activation=theano.tensor.nnet.relu

    )

    layer4 = LeNetConvPoolLayer(
        rng,
        input=layer3.output,
        image_shape=(batch_size, nkerns[3], 24, 24),
        filter_shape=(nkerns[4], nkerns[3], 3, 3),
        poolsize=(2, 2),
        activation=theano.tensor.nnet.relu
    )

    layer5_input = layer4.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer5 = Fully_Connected_Dropout(
        rng,
        input=layer5_input,
        n_in=nkerns[4] * 11 * 11,
        n_out=4096,
        is_train=is_train,
        activation=theano.tensor.nnet.relu
    )

    layer6 = Fully_Connected_Dropout(
        rng,
        input=layer5.output,
        n_in=4096,
        n_out=4096,
        is_train=is_train,
        activation=theano.tensor.nnet.relu
    )

    layer7 = FullyConnected(
        rng,
        input=layer6.output,
        n_in=4096,
        n_out=2000,
        activation=theano.tensor.nnet.relu
    )


    

    layer8 = LogisticRegression(input=layer7.output, n_in=2000, n_out=2)



    salidas_capa8 = theano.function(
        [index],
        layer8.y_pred,
        on_unused_input='ignore',
        givens={
            x: test_set_x_rgb[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            is_train: numpy.cast['int32'](0)
        }
    )

  
    salidas_probabilidad = theano.function(
        [index],
        layer8.p_y_given_x,
        on_unused_input='ignore',

        givens={
            x: test_set_x_rgb[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            is_train: numpy.cast['int32'](0)
        }
    )


    salidas_capa7_test = theano.function(
        [index],
        layer7.output,
        on_unused_input='ignore',
        givens={
            x: test_set_x_rgb[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            is_train: numpy.cast['int32'](0)
        }
    )

 
    salidas_capa7_train = theano.function(
        [index],
        layer7.output,
        on_unused_input='ignore',
        givens={
            x: train_set_x_rgb[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            is_train: numpy.cast['int32'](1)
        }
    )

   

    # the cost we minimize during training is the NLL of the model
    cost = layer8.negative_log_likelihood(y)


    test_model = theano.function(
        [index],
        layer8.errors(y),
        givens={
            x: test_set_x_rgb[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            is_train: numpy.cast['int32'](0)
        }
    )

    validate_model = theano.function(
        [index],
        layer8.errors(y),
        givens={
            x: valid_set_x_rgb[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size],
            is_train: numpy.cast['int32'](0)

        }
    )

    


    params = layer0.params + layer1.params + layer2.params + layer3.params + layer4.params + layer5.params + layer6.params + layer7.params + layer8.params


    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)



    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)]


    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x_rgb[index * batch_size: (index + np.cast['int32'](1)) * batch_size],
            y: train_set_y[index * batch_size: (index + np.cast['int32'](1)) * batch_size],
            is_train: np.cast['int32'](1)
        }
    )

   

    ###################
    # TRAIN NIR MODEL #
    ###################
    print ('... training')
    print (' ')
    # early-stopping parameters
    patience = 100000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is
    # considered significant
    validation_frequency = min(n_train_batches, patience / 2)

    print("patience: %d" % patience)
    print("patience_increase: %d" % patience_increase)
    print("improvement threshold: %d" % improvement_threshold)
    print("validation_frequency: %d" % validation_frequency)
    print (' ')


    best_validation_loss = numpy.inf
    best_iter = 0
    start_time = timeit.default_timer()
    error_epoch = []
    lista_coste = []
    epoch = 0
    done_looping = False

    print ('n_train_batches', n_train_batches)
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)

            cost_ij = train_model(minibatch_index)
            lista_coste.append(cost_ij)

            if (iter + 1) % validation_frequency == 0:

                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%, cost %f' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100., cost_ij))

                error_epoch.append(this_validation_loss * 100)


                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * \
                            improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    w0_test = layer0.W.get_value()
                    b0_test = layer0.b.get_value()

                    w1_test = layer1.W.get_value()
                    b1_test = layer1.b.get_value()

                    w2_test = layer2.W.get_value()
                    b2_test = layer2.b.get_value()

                    w3_test = layer3.W.get_value()
                    b3_test = layer3.b.get_value()

                    w4_test = layer4.W.get_value()
                    b4_test = layer4.b.get_value()

                    w5_test = layer5.W.get_value()
                    b5_test = layer5.b.get_value()

                    w6_test = layer6.W.get_value()
                    b6_test = layer6.b.get_value()

                    w7_test = layer7.W.get_value()
                    b7_test = layer7.b.get_value()
                    w8_test = layer8.W.get_value()
                    b8_test = layer8.b.get_value()

                    #Prepare input data for training
                    input_TrainClass = []

                    # sal_capa2 = [salidas_capa2_train(i) for i in xrange(n_train_batches)]
                    sal_capa2 = [salidas_capa7_train(i) for i in range(n_train_batches)]
                    for i in sal_capa2:
                        for j in i:
                            input_TrainClass.append(j)

                    #input_TrainClass_concatenated = np.concatenate([input_TrainClass,input_TrainClass_1], axis=1)



    ###############################
    ###    TESTING MODEL        ###
    ###############################

    layer0.W.set_value(w0_test)
    layer0.b.set_value(b0_test)

    layer1.W.set_value(w1_test)
    layer1.b.set_value(b1_test)

    layer2.W.set_value(w2_test)
    layer2.b.set_value(b2_test)

    layer3.W.set_value(w3_test)
    layer3.b.set_value(b3_test)

    layer4.W.set_value(w4_test)
    layer4.b.set_value(b4_test)

    layer5.W.set_value(w5_test)
    layer5.b.set_value(b5_test)

    layer6.W.set_value(w6_test)
    layer6.b.set_value(b6_test)

    layer7.W.set_value(w7_test)
    layer7.b.set_value(b7_test)
    layer8.W.set_value(w8_test)
    layer8.b.set_value(b8_test)

   
    y_pred_junto = []
    y_prob_junto = []

    input_test = []

    sal_capa7 = [salidas_capa7_test(i) for i in range(n_test_batches)]

    for i in sal_capa7:
        for j in i:
            input_test.append(j)

    # test it on the test set
    test_losses = [test_model(i) for i in range(n_test_batches)]
    test_score = numpy.mean(test_losses)

    print ('NIR model test losses:', test_score)

    plt.clf()
    plt.plot(error_epoch)
    plt.ylabel('error')
    plt.xlabel('epoch')
    plt.savefig(out_path+'error_nir.png')

    plt.clf()
    plt.plot(lista_coste)
    plt.ylabel('cost_ij')
    plt.xlabel('iteration')
    plt.savefig(out_path+'cost_nir.png')
    
    print ('NIR best model obtained at iter:', best_iter,'whit a valid loss:', best_validation_loss)     

    return input_TrainClass, input_test
