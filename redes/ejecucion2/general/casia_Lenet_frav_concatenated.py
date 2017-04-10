# En este archivo de python se va a utilizar la base de datos de FRAV y Casia de imagenes  y se va a seguir la configuracion de la CNN del paper 'Learn convolutional neural network for face anti-spoofing'
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy
import timeit
from pylab import *
from logistic_sgd import LogisticRegression
from layers import *
import sys
import theano
import theano.tensor
import auxiliar_functions
import pickle
import os, argparse
from casia_Lenet3_RGB import evaluate_lenet5_RGB
from casia_Lenet3_NIR import evaluate_lenet5_NIR
from softmax_concatenated import *
from classifiers import *
from ISOmetrics import *

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


    print (numpy.array(train_set_x_rgb).shape, numpy.array(test_set_x_rgb).shape, numpy.array(valid_set_x_rgb).shape)
    print (numpy.array(y_train).shape, numpy.array(y_test).shape, numpy.array(y_val).shape)

    print((train_set_x_rgb[0]).shape)

    data = [train_set_x_rgb, test_set_x_rgb, valid_set_x_rgb, y_train, y_test, y_val]
    data1 = [train_set_x_nir, test_set_x_nir, valid_set_x_nir, y_train, y_test, y_val]
    # Get train, valid and test features of layer 7
    RGB_train_x, RGB_test_x, RGB_valid_x = evaluate_lenet5_RGB(learning_rate, n_epochs, nkerns, batch_size, data, out_path)
    NIR_train_x, NIR_test_x, NIR_valid_x = evaluate_lenet5_NIR(learning_rate, n_epochs, nkerns, batch_size, data1, out_path)
 	
    # Concatenate the RGB and NIR outputs
    input_TrainClass_concatenated = np.concatenate([RGB_train_x, NIR_train_x], axis=1)
    input_TestClass_concatenated = np.concatenate([RGB_test_x, NIR_test_x], axis=1)
    input_ValidClass_concatenated = np.concatenate([RGB_valid_x, NIR_valid_x], axis=1)

	# Get y as the same length
    y_train = y_train[0:len(input_TrainClass_concatenated)]
    y_test = y_test[0:len(input_TestClass_concatenated)]
    y_valid = y_val[0:len(input_ValidClass_concatenated)]




      ########## SVM ###########

    SVM_pred, SVM_pred_prob, scores_SVM = SVMClas(input_TrainClass_concatenated, y_train, input_TestClass_concatenated, y_test, 'rbf')
    print ('SVM RBF scores:', scores_SVM)
    TP, TN, FP, FN = auxiliar_functions.analize_results(y_test, SVM_pred, SVM_pred_prob, out_path+'SVM_RBF-')
    DETCurve(y_test, SVM_pred_prob[:,1], 1, out_path+'SVM_RBF-')
    metric(SVM_pred_prob[:,1], SVM_pred, y_test, 1, out_path+'SVM_RBF-')

    ########## SVM LINEAR ###########

    SVM_pred, SVM_pred_prob, scores_SVM = SVMClas(input_TrainClass_concatenated, y_train, input_TestClass_concatenated, y_test, 'linear')

    print ('SVM linear scores:', scores_SVM)
    TP, TN, FP, FN =auxiliar_functions.analize_results(y_test, SVM_pred, SVM_pred_prob, out_path+'SVM_LINEAR-')
    DETCurve(y_test, SVM_pred_prob[:,1], 1, out_path+'SVM_linear-')
    metric(SVM_pred_prob[:,1], SVM_pred, y_test, 1, out_path+'SVM_linear-')

    ############  KNN  ##############

    knn_pred, knn_pred_prob, scores_knn = KNNClas(input_TrainClass_concatenated, y_train, input_TestClass_concatenated, y_test)

    print('KNN mean accuracy: ', scores_knn)
    TP, TN, FP, FN =auxiliar_functions.analize_results(y_test, knn_pred, knn_pred_prob, out_path+'KNN-')
    DETCurve(y_test, knn_pred_prob[:,1], 1, out_path+'KNN-')
    metric(knn_pred_prob[:,1], knn_pred, y_test, 1, out_path+'KNN-')

    ########## DECISION TREE ###########
    tree_pred, tree_pred_prob, scores_tree = DecisionTreeClas(input_TrainClass_concatenated, y_train, input_TestClass_concatenated, y_test)
    print('Decision Tree mean accuracy: ', scores_tree)
    TP, TN, FP, FN = auxiliar_functions.analize_results(y_test, tree_pred, tree_pred_prob, out_path + 'DecisionTree-')
    DETCurve(y_test, tree_pred_prob[:,1], 1, out_path+'TREE-')
    metric(tree_pred_prob[:,1], tree_pred, y_test, 1, out_path+'TREE-')


    ########### SOFTMAX ############

    print('----------SOFTMAX ---------')
    learning_rates = [0.0001, 0.005, 0.001, 0.05, 0.01]
    iterarions = 400
    data = [input_TrainClass_concatenated, input_ValidClass_concatenated, input_TestClass_concatenated , y_train, y_val, y_test]
    validation_losses = np.zeros(len(learning_rates))
    test_scores = np.zeros(len(learning_rates))
    iters = np.zeros(len(learning_rates))
    Softmax_predictions = []
    Softmax_probabilities = []
    for i, lr in enumerate(learning_rates):
        validation_losses[i], test_scores[i], iters[i], predictions_aux, probabilities_aux = sgd_optimization(lr, 400, batch_size, data,len(input_TrainClass_concatenated[0]), 1)
        Softmax_predictions.append(predictions_aux)
        Softmax_probabilities.append(probabilities_aux)
    indice_softmax = np.argmax(test_scores)
    auxiliar_functions.analize_results(y_test, Softmax_predictions[indice_softmax], Softmax_probabilities[indice_softmax], out_path+'PCA_Softmax-')
    metric(Softmax_predictions[indice_softmax], Softmax_probabilities[indice_softmax], y_test, 1, out_path+'PCA_Softmax-')
    print('best softmax PCA scores with a learning rate of', learning_rates[indice_softmax], 'best validation score:', validation_losses[indice_softmax],'at iteration', iters[indice_softmax], 'and test loss:', test_scores[indice_softmax])


    ############## PCA ################

    X_train_after_PCA, X_test_after_PCA, X_valid_after_PCA = PCAClas(input_TrainClass_concatenated, y_train, input_TestClass_concatenated, y_test, input_ValidClass_concatenated)

    ########## PCA SVM ###########

    SVM_pred, SVM_pred_prob, scores_SVM = SVMClas(X_train_after_PCA, y_train, X_test_after_PCA, y_test, 'rbf')
    print ('SVM RBF scores:', scores_SVM)
    TP, TN, FP, FN =auxiliar_functions.analize_results(y_test, SVM_pred, SVM_pred_prob, out_path + 'PCA_SVM_RBF-')
    DETCurve(y_test, SVM_pred_prob[:,1], 1,out_path+'PCA_SVM_RBF-')
    metric(SVM_pred_prob[:,1], SVM_pred, y_test, 1, out_path+'PCA_SVM_RBF-')

    ########## PCA SVM LINEAR ###########

    SVM_pred, SVM_pred_prob, scores_SVM = SVMClas(X_train_after_PCA, y_train, X_test_after_PCA, y_test, 'linear')

    print ('SVM linear scores:', scores_SVM)
    TP, TN, FP, FN = auxiliar_functions.analize_results(y_test, SVM_pred, SVM_pred_prob, out_path + 'PCA_SVM_LINEAR-')
    DETCurve(y_test, SVM_pred_prob[:,1], 1, out_path+'PCA_SVM_linear-')
    metric(SVM_pred_prob[:,1], SVM_pred, y_test, 1, out_path+'PCA_SVM_Linear-')

    ############  PCA KNN  ##############

    knn_pred, knn_pred_prob, scores_knn = KNNClas(X_train_after_PCA, y_train, X_test_after_PCA, y_test)
    print('KNN mean accuracy: ', scores_knn)
    TP, TN, FP, FN = auxiliar_functions.analize_results(y_test, knn_pred, knn_pred_prob, out_path + 'PCA_KNN-')
    DETCurve(y_test, knn_pred_prob[:,1], 1, out_path+'PCA_KNN-')
    metric(knn_pred_prob[:,1], knn_pred, y_test, 1, out_path+'PCA_KNN-')

    ########## DECISION TREE ###########
    tree_pred, tree_pred_prob, scores_tree = DecisionTreeClas(X_train_after_PCA, y_train, X_test_after_PCA, y_test)
    print('PCA Decision Tree mean accuracy: ', scores_tree)
    TP, TN, FP, FN = auxiliar_functions.analize_results(y_test, tree_pred, tree_pred_prob, out_path + 'PCA_DecisionTree-')
    DETCurve(y_test, tree_pred_prob[:,1], 1, out_path+'PCA_TREE-')
    metric(tree_pred_prob[:,1], tree_pred, y_test, 1, out_path+'PCA_TREE-')

    ########### SOFTMAX + PCA ############
    print('----------SOFTMAX + PCA---------')
    learning_rates = [0.0001, 0.005, 0.001, 0.05, 0.01]
    iterarions = 400
    data = [X_train_after_PCA, X_valid_after_PCA, X_test_after_PCA , y_train, y_val, y_test]
    validation_losses = np.zeros(len(learning_rates))
    test_scores = np.zeros(len(learning_rates))
    iters = np.zeros(len(learning_rates))
    Softmax_predictions = []
    Softmax_probabilities = []
    for i, lr in enumerate(learning_rates):
        validation_losses[i], test_scores[i], iters[i], predictions_aux, probabilities_aux = sgd_optimization(lr, 400, batch_size, data, len(X_train_after_PCA[0]), 1)
        Softmax_predictions.append(predictions_aux)
        Softmax_probabilities.append(probabilities_aux)
    indice_softmax = np.argmax(test_scores)
    auxiliar_functions.analize_results(y_test, Softmax_predictions[indice_softmax], Softmax_probabilities[indice_softmax], out_path+'PCA_Softmax-')
    metric(Softmax_predictions[indice_softmax], Softmax_probabilities[indice_softmax], y_test, 1, out_path+'PCA_Softmax-')
    print('best softmax PCA scores with a learning rate of', learning_rates[indice_softmax], 'best validation score:', validation_losses[indice_softmax],'at iteration', iters[indice_softmax], 'and test loss:', test_scores[indice_softmax])

    ##############    LDA   ###################
    X_train_after_LDA, X_test_after_LDA, X_valid_after_LDA = LDAClas(input_TrainClass_concatenated, y_train,input_TestClass_concatenated, y_test, input_ValidClass_concatenated)

    ########## LDA SVM ###########

    SVM_pred, SVM_pred_prob, scores_SVM = SVMClas(X_train_after_LDA, y_train, X_test_after_LDA, y_test, 'rbf')
    print ('SVM RBF scores:', scores_SVM)
    TP, TN, FP, FN =auxiliar_functions.analize_results(y_test, SVM_pred, SVM_pred_prob, out_path + 'LDA_SVM_RBF-')
    DETCurve(y_test, SVM_pred_prob[:,1], 1,out_path+'LDA_SVM_RBF-')
    metric(SVM_pred_prob[:,1], SVM_pred, y_test, 1, out_path+'LDA_SVM_RBF-')


    ########## LDA SVM LINEAR ###########

    SVM_pred, SVM_pred_prob, scores_SVM = SVMClas(X_train_after_LDA, y_train, X_test_after_LDA, y_test, 'linear')

    print ('SVM linear scores:', scores_SVM)
    TP, TN, FP, FN = auxiliar_functions.analize_results(y_test, SVM_pred, SVM_pred_prob, out_path + 'LDA_SVM_LINEAR-')
    DETCurve(y_test, SVM_pred_prob[:,1], 1, out_path+'LDA_SVM_linear-')
    metric(SVM_pred_prob[:,1], SVM_pred, y_test, 1, out_path+'LDA_SVM_linear-')

    ############  LDA KNN  ##############

    knn_pred, knn_pred_prob, scores_knn = KNNClas(X_train_after_LDA, y_train, X_test_after_LDA, y_test)
    print('KNN mean accuracy: ', scores_knn)
    TP, TN, FP, FN = auxiliar_functions.analize_results(y_test, knn_pred, knn_pred_prob, out_path + 'LDA_KNN-')
    DETCurve(y_test, knn_pred_prob[:,1], 1, out_path+'LDA_KNN-')
    metric(knn_pred_prob[:,1], knn_pred, y_test, 1, out_path+'LDA_KNN-')

    ##########  LDA DECISION TREE ###########
    tree_pred, tree_pred_prob, scores_tree = DecisionTreeClas(X_train_after_LDA, y_train, X_test_after_LDA, y_test)
    print('LDA-Tree mean accuracy: ', scores_tree)
    TP, TN, FP, FN = auxiliar_functions.analize_results(y_test, tree_pred, tree_pred_prob, out_path + 'LDA_DecisionTree-')
    DETCurve(y_test, tree_pred_prob[:,1], 1, out_path+'LDA_TREE-')
    metric(tree_pred_prob[:,1], tree_pred, y_test, 1, out_path+'LDA_TREE-')

    ########### SOFTMAX + LDA ############
    print('----------SOFTMAX + LDA---------')
    learning_rates = [0.0001, 0.005, 0.001, 0.05, 0.01]
    iterarions = 400
    data = [X_train_after_LDA, X_valid_after_LDA, X_test_after_LDA , y_train, y_val, y_test]
    validation_losses = np.zeros(len(learning_rates))
    test_scores = np.zeros(len(learning_rates))
    iters = np.zeros(len(learning_rates))
    Softmax_predictions = []
    Softmax_probabilities = []
    for i, lr in enumerate(learning_rates):
        validation_losses[i], test_scores[i], iters[i], predictions_aux, probabilities_aux = sgd_optimization(lr, 400, batch_size, data, len(X_train_after_LDA[0]), 1)
        Softmax_predictions.append(predictions_aux)
        Softmax_probabilities.append(probabilities_aux)
    indice_softmax = np.argmax(test_scores)
    auxiliar_functions.analize_results(y_test, Softmax_predictions[indice_softmax], Softmax_probabilities[indice_softmax], out_path+'LDA_Softmax-')
    metric(Softmax_predictions[indice_softmax], Softmax_probabilities[indice_softmax], y_test, 1, out_path+'LDA_Softmax-')
    print('best softmax LDA scores with a learning rate of', learning_rates[indice_softmax], 'best validation score:', validation_losses[indice_softmax],'at iteration', iters[indice_softmax], 'and test loss:', test_scores[indice_softmax])



   ########  FIN CLASIFICADORES #########


    end_time = timeit.default_timer()
    print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.)))


    sys.stdout = orig_stdout
    f.close()


if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
