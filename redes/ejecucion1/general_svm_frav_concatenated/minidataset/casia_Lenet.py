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


def evaluate_lenet5(learning_rate=0.01, n_epochs=400, nkerns=[9, 2, 3, 8, 2], batch_size=20):

    argumento = argparse.ArgumentParser()

    argumento.add_argument('-i', '--i', required=True, help="path where the pkl dabase file is")
    argumento.add_argument("-o", "--o", required=True,help="path where save the output")

    # Se le asigna el primer argumento a la ruta de entrada
    args = vars(argumento.parse_args())
    in_path = args['i']

    # Se le asigna el segundo argumento ala ruta de salida
    out_path = args['o']

    orig_stdout = sys.stdout
    f = file(out_path+'out.txt', 'w')
    sys.stdout = f

    print ('In this file are the results of using casia architecture and FRAV image database')
    print ('In the architecture conv, pool, response normalization,fully connect, dropout and softmax layers are used with relu. No strides are used')
    print ('The configuration of the net is learning rate = ', learning_rate, 'numero de epochs =',n_epochs, 'numero de kernels', nkerns, 'batch_size =',batch_size')
    print ('Early stop has been deleted')
    print('For training has been used softmax classifier and for testing softmax and SVM')
    print ('In this example, two classes are going to be used, class 0 for real users and class 1 for attacks')

    print ('Start reading the data...')
    rng = numpy.random.RandomState(123456)


    # open_file2 = open('C:\Users\FRAV\Desktop\Beatriz\FRAv_casia_ImageNet\data_casia_all.pkl', 'rb')
    open_file2 = open(in_path, 'rb')
    [train_index, test_index, validate_index, train_set_x_rgb, test_set_x_rgb, valid_set_x_rgb, train_set_x_nir, test_set_x_nir, valid_set_x_nir, y_train,  y_test, y_val] = pickle.load(open_file2)
    open_file2.close()

    train_set_x_rgb = train_set_x_rgb[0:20]
    test_set_x_rgb = test_set_x_rgb[0:20]
    valid_set_x_rgb= valid_set_x_rgb[0:20]

    train_set_x_nir = train_set_x_nir[0:20]
    test_set_x_nir= test_set_x_nir[0:20]
    valid_set_x_nir= valid_set_x_nir[0:20]

    y_train = y_train[0:20]
    y_test = y_test[0:20]
    y_val = y_val[0:20]

    print (numpy.array(train_set_x_rgb).shape, numpy.array(test_set_x_rgb).shape, numpy.array(valid_set_x_rgb).shape)
    print (numpy.array(y_train).shape, numpy.array(y_test).shape, numpy.array(y_val).shape)

    print((train_set_x_rgb[0]).shape)

    train_set_x_rgb = theano.shared(numpy.array(train_set_x_rgb, dtype= 'float32'), borrow=True)
    test_set_x_rgb = theano.shared(numpy.array(test_set_x_rgb, dtype= 'float32'), borrow=True)
    valid_set_x_rgb = theano.shared(numpy.array(valid_set_x_rgb, dtype= 'float32'), borrow=True)

    train_set_x_nir = theano.shared(numpy.array(train_set_x_nir, dtype= 'float32'), borrow=True)
    test_set_x_nir = theano.shared(numpy.array(test_set_x_nir, dtype= 'float32'), borrow=True)
    valid_set_x_nir = theano.shared(numpy.array(valid_set_x_nir, dtype= 'float32'), borrow=True)

    train_set_y = theano.shared(numpy.array(y_train, dtype='int32'), borrow=True)
    test_set_y = theano.shared(numpy.array(y_test, dtype='int32'), borrow=True)
    valid_set_y = theano.shared(numpy.array(y_val,dtype='int32'), borrow=True)


    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x_nir.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x_nir.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x_nir.get_value(borrow=True).shape[0]

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
    x1 = T.matrix('x1')# the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
    is_train = T.iscalar('is_train')  # To differenciate between train and test

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print ('... building the model')


    ## RGB MODEL ##


    layer0_input = x.reshape((batch_size, 3, 128, 128))

    layer0 = LeNetConvPoolLRNLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 128, 128),
        filter_shape=(nkerns[0], 3, 11, 11),
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


    ## NIR MODEL ##

    layer0_input_1 = x1.reshape((batch_size, 1, 128, 128))

    layer0_1 = LeNetConvPoolLRNLayer(
        rng,
        input=layer0_input_1,
        image_shape=(batch_size, 1, 128, 128),
        filter_shape=(nkerns[0], 1, 11, 11),
        stride=(1, 1),
        lrn=True,
        poolsize=(2, 2),
        activation=theano.tensor.nnet.relu
    )

    layer1_1 = LeNetConvPoolLRNLayer(
        rng,
        input=layer0_1.output,
        # image_shape=(batch_size, nkerns[0], 27, 27),
        image_shape=(batch_size, nkerns[0], 59, 59),
        filter_shape=(nkerns[1], nkerns[0], 4, 4),
        lrn=True,
        poolsize=(2, 2),
        activation=theano.tensor.nnet.relu
    )

    layer2_1 = LeNetConvPoolLayer(
        rng,
        input=layer1_1.output,
        image_shape=(batch_size, nkerns[1], 28, 28),
        filter_shape=(nkerns[2], nkerns[1], 3, 3),
        poolsize=(1, 1),
        activation=theano.tensor.nnet.relu

    )

    layer3_1 = LeNetConvPoolLayer(
        rng,
        input=layer2_1.output,
        image_shape=(batch_size, nkerns[2], 26, 26),
        filter_shape=(nkerns[3], nkerns[2], 3, 3),
        poolsize=(1, 1),
        activation=theano.tensor.nnet.relu

    )

    layer4_1 = LeNetConvPoolLayer(
        rng,
        input=layer3_1.output,
        image_shape=(batch_size, nkerns[3], 24, 24),
        filter_shape=(nkerns[4], nkerns[3], 3, 3),
        poolsize=(2, 2),
        activation=theano.tensor.nnet.relu
    )

    layer5_input_1 = layer4_1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer5_1 = Fully_Connected_Dropout(
        rng,
        input=layer5_input_1,
        n_in=nkerns[4] * 11 * 11,
        n_out=4096,
        is_train=is_train,
        activation=theano.tensor.nnet.relu
    )

    layer6_1 = Fully_Connected_Dropout(
        rng,
        input=layer5_1.output,
        n_in=4096,
        n_out=4096,
        is_train=is_train,
        activation=theano.tensor.nnet.relu
    )

    layer7_1 = FullyConnected(
        rng,
        input=layer6_1.output,
        n_in=4096,
        n_out=2000,
        activation=theano.tensor.nnet.relu
    )

    # Concatenate the ouputs of both models
    layer8_input = T.concatenate([layer7.output,layer7_1.output], axis=1)


    layer8 = LogisticRegression(input=layer8_input, n_in=4000, n_out=2)


    salidas_capa8 = theano.function(
        [index],
        layer8.y_pred,
        on_unused_input='ignore',
        givens={
            x: test_set_x_rgb[index * batch_size: (index + 1) * batch_size],
            x1: test_set_x_nir[index * batch_size: (index + 1) * batch_size],
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
            x1: test_set_x_nir[index * batch_size: (index + 1) * batch_size],
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
            x1: test_set_x_nir[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            is_train: numpy.cast['int32'](0)
        }
    )

    salidas_capa7_test_1 = theano.function(
        [index],
        layer7_1.output,
        on_unused_input='ignore',
        givens={
            x: test_set_x_rgb[index * batch_size: (index + 1) * batch_size],
            x1: test_set_x_nir[index * batch_size: (index + 1) * batch_size],
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
            x1: train_set_x_nir[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            is_train: numpy.cast['int32'](1)
        }
    )

    salidas_capa7_train_1 = theano.function(
        [index],
        layer7_1.output,
        on_unused_input='ignore',
        givens={
            x: train_set_x_rgb[index * batch_size: (index + 1) * batch_size],
            x1: train_set_x_nir[index * batch_size: (index + 1) * batch_size],
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
            x1: test_set_x_nir[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            is_train: numpy.cast['int32'](0)
        }
    )

    validate_model = theano.function(
        [index],
        layer8.errors(y),
        givens={
            x: valid_set_x_rgb[index * batch_size: (index + 1) * batch_size],
            x1: valid_set_x_nir[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size],
            is_train: numpy.cast['int32'](0)

        }
    )

    params = layer0.params + layer1.params + layer2.params + layer3.params + layer4.params + layer5.params + layer6.params + layer7.params + layer0_1.params + layer1_1.params + layer2_1.params + layer3_1.params + layer4_1.params + layer5_1.params + layer6_1.params + layer7_1.params + layer8.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)


    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
        ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x_rgb[index * batch_size: (index + np.cast['int32'](1)) * batch_size],
            x1: train_set_x_nir[index * batch_size: (index + np.cast['int32'](1)) * batch_size],
            y: train_set_y[index * batch_size: (index + np.cast['int32'](1)) * batch_size],
            is_train: np.cast['int32'](1)
        }
    )


    ###############
    # TRAIN MODEL #
    ###############
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

                    #Prepare input data for training
                    input_TrainClass = []
                    input_TrainClass_1 = []

                    # sal_capa2 = [salidas_capa2_train(i) for i in xrange(n_train_batches)]
                    sal_capa2 = [salidas_capa7_train(i) for i in range(n_train_batches)]
                    for i in sal_capa2:
                        for j in i:
                            input_TrainClass.append(j)

                    sal_capa2_1 = [salidas_capa7_train_1(i) for i in range(n_train_batches)]
                    for i in sal_capa2_1:
                        for j in i:
                            input_TrainClass_1.append(j)

                    input_TrainClass_concatenated = np.concatenate([input_TrainClass,input_TrainClass_1], axis=1)


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

    y_pred_junto = []
    y_prob_junto = []

    input_test = []
    input_test_1 = []

    sal_capa7 = [salidas_capa7_test(i) for i in range(n_test_batches)]

    for i in sal_capa7:
        for j in i:
            input_test.append(j)

    sal_capa7_1 = [salidas_capa7_test_1(i) for i in range(n_test_batches)]

    for i in sal_capa7_1:
        for j in i:
            input_test_1.append(j)


    input_TestClass_concatenated = np.concatenate([input_test, input_test_1], axis=1)

    ##### SVM #####
    svm_scores = []
    C_range = [0.001,0.005 ,0.01, 0.05, 0.1, 0.5, 1, 2,3, 5, 10]

    y_train = y_train[0:len(input_TrainClass_concatenated)]
    y_test = y_test[0:len(input_test)]

    for Ci in C_range:
        SVMcla = SVC(C=Ci, kernel='rbf')
        scores = cross_val_score(SVMcla, input_TrainClass_concatenated, y_train, cv=5,
                                 scoring='accuracy')  # Con neg_log_loss el predict tiene que ser con probabilidad
        svm_scores.append(scores.mean())

    optimaC = C_range[svm_scores.index(max(svm_scores))]
    print('optimaC', optimaC)
    SVM2 = SVC(C = optimaC, kernel='rbf', probability= True)
    SVM2.fit(X = input_TrainClass_concatenated,y = y_train)

    SVM_pred = SVM2.predict(input_TestClass_concatenated)
    SVM_pred_prob = SVM2.predict_proba(input_TestClass_concatenated)
    scores_SVM = SVM2.score(input_TestClass_concatenated,y_test)

    # test it on the test set
    test_losses = [test_model(i) for i in range(n_test_batches)]
    test_score = numpy.mean(test_losses)

    ##### SOFTMAX #####
    for i in range(n_test_batches):
        y_pred_test = salidas_capa8(i)
        y_probabilidad = salidas_probabilidad(i)

        for j in y_pred_test:
            y_pred_junto.append(j)

        for j in y_probabilidad:
            y_prob_junto.append(j[0])

    print ('softmax')
    print((' test error of best model %f %%') % (test_score * 100.))

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))



    plt.clf()
    plt.plot(error_epoch)
    plt.ylabel('error')
    plt.xlabel('epoch')
    plt.savefig(out_path+'error.png')

    plt.clf()
    plt.plot(lista_coste)
    plt.ylabel('cost_ij')
    plt.xlabel('iteration')
    plt.savefig(out_path+'cost.png')


    print ('SVM scores:', scores_SVM)
    auxiliar_functions.analize_results(y_test, SVM_pred, SVM_pred_prob, out_path)

    print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.)))


    sys.stdout = orig_stdout
    f.close()


if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
