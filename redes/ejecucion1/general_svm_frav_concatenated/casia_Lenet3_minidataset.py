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
from casia_Lenet3_RGB import evaluate_lenet5_RGB
from casia_Lenet3_NIR import evaluate_lenet5_NIR
from softmax_concatenated import *

#nkerns=[96, 256, 386, 384, 256]


def evaluate_lenet5(learning_rate=0.01, n_epochs=400, nkerns=[96, 256, 386, 384, 256], batch_size=20):
    start_time = timeit.default_timer()

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
    print ('The configuration of the net is learning_rate=0.001, n_epochs=400, nkerns=[96, 256, 386, 384, 256], batch_size=20')
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
    valid_set_x_rgb = valid_set_x_rgb[0:20]
    train_set_x_nir = train_set_x_nir[0:20]
    test_set_x_nir = test_set_x_nir[0:20]
    valid_set_x_nir = valid_set_x_nir[0:20]

    y_train = y_train[0:20]
    y_val = y_val[0:20]
    y_test = y_test[0:20]

    print (numpy.array(train_set_x_rgb).shape, numpy.array(test_set_x_rgb).shape, numpy.array(valid_set_x_rgb).shape)
    print (numpy.array(y_train).shape, numpy.array(y_test).shape, numpy.array(y_val).shape)

    print((train_set_x_rgb[0]).shape)

    data = [train_set_x_rgb, test_set_x_rgb, valid_set_x_rgb, y_train, y_test, y_val]
    data1 = [train_set_x_nir, test_set_x_nir, valid_set_x_nir, y_train, y_test, y_val]

    RGB_train_x, RGB_test_x, RGB_valid_x = evaluate_lenet5_RGB(learning_rate, n_epochs, nkerns, batch_size, data, out_path)
    NIR_train_x, NIR_test_x, NIR_valid_x = evaluate_lenet5_NIR(learning_rate, n_epochs, nkerns, batch_size, data1, out_path)
 
    input_TrainClass_concatenated = np.concatenate([RGB_train_x, NIR_train_x], axis=1)
    input_TestClass_concatenated = np.concatenate([RGB_test_x, NIR_test_x], axis=1)
    input_ValidClass_concatenated = np.concatenate([RGB_valid_x, NIR_valid_x], axis=1)
    ##### SVM #####
    svm_scores = []
    C_range = [0.001,0.005 ,0.01, 0.05, 0.1, 0.5, 1, 2,3, 5, 10]

    y_train = y_train[0:len(input_TrainClass_concatenated)]
    y_test = y_test[0:len(input_TestClass_concatenated)]
    y_val = y_val[0:len(input_ValidClass_concatenated)]

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

    print ('SVM scores:', scores_SVM)
    auxiliar_functions.analize_results(y_test, SVM_pred, SVM_pred_prob, out_path+'SVM-')

    learning_rates = [0.0001, 0.005, 0.001, 0.05, 0.01]
    iterarions = 400
    
    data = [input_TrainClass_concatenated, input_ValidClass_concatenated, input_TestClass_concatenated, y_train, y_val, y_test]
    validation_losses = np.zeros(len(learning_rates))
    test_scores = np.zeros(len(learning_rates))
    iters = np.zeros(len(learning_rates))
    Softmax_predictions = []
    Softmax_probabilities = []
    for i, lr in enumerate(learning_rates):
        validation_losses[i], test_scores[i], iters[i], predictions_aux, probabilities_aux = sgd_optimization(lr, 400, batch_size, data, 4000, 1)
        Softmax_predictions.append(predictions_aux)
        Softmax_probabilities.append(probabilities_aux)
    indice_softmax = np.argmax(test_scores)
    print('SOFTMAX')
    auxiliar_functions.analize_results(y_test, Softmax_predictions[indice_softmax], Softmax_probabilities[indice_softmax], out_path+'Softmax-')
    print('best softmax scores with a learning rate of', learning_rates[indice_softmax], 'best validation score;', validation_losses[indice_softmax], 'at iteration', iters[indice_softmax], 'and test loss:', test_scores[indice_softmax])


    end_time = timeit.default_timer()

    print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.)))


    sys.stdout = orig_stdout
    f.close()


if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
