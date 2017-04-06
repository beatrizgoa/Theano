import numpy as np
import random
import cv2
import pickle
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from itertools import cycle


def randomize(Xtrain, ytrain, Xtest, ytest, seed):

    total_train = len(Xtrain)
    print ('Xtrain_len')
    print (len(Xtrain))
    index_train = np.arange(total_train)


    random.shuffle(index_train, lambda: 0.5)
    random.shuffle(index_train, lambda: seed)


    validate_stop = int(round(total_train * 0.3))

    validate_index = index_train[0:validate_stop]
    train_index = index_train[validate_stop:total_train]


    X_train = []
    y_train = []
    X_validate = []
    y_validate = []


    for i in train_index:
        yaux = np.array(ytrain[i])
        Xaux = np.array(Xtrain[i])
        X_train.append(Xaux)
        y_train.append(yaux)

    for i in validate_index:
        yaux = np.array(ytrain[i])
        Xaux = np.array(Xtrain[i])
        X_validate.append(Xaux)
        y_validate.append(yaux)


    data = [train_index, validate_index, X_train, Xtest, y_train, ytest, X_validate, y_validate]

    # save_file2 = open('/home/beaa/Escritorio/Theano/results/casia_videos/data_read.pkl', 'wb')
    # pickle.dump(data, save_file2, -1)
    # save_file2.close()

    Xtrain = 0
    ytrain = 0

    return train_index, validate_index, X_train, y_train, X_validate, y_validate





def procesing(frame):
    # Face viola-jones detector is defined
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Spacial scales ued in casia
    spacial_scales = [1, 1.4, 1.8, 2.2, 2.6]
    aux_X = []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Get image to gray_Scale
    [row_size, col_size, w_size] = frame.shape  # Get shape pf images
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Get faces

    for (x, y, w, h) in faces:
        face_cropped = frame[y:y + h, x:x + w]  # The faces are cropped

        if (w*h)>(row_size*col_size/5):

            for scale in spacial_scales:
                new_h = round(h * scale)
                diference = new_h - h
                sum = round(diference / 2)

                new_w = round(w * scale)
                diference_w = new_w - w
                sum_w = round(diference_w / 2)

                arr = (y - sum)
                abj = (y - sum) + new_h
                izq = (x - sum_w)
                dcha = (x - sum_w + w) + w

                if izq < 0:
                    izq = 0
                if dcha > col_size:
                    dcha = col_size

                if arr < 0:
                    arr = 0
                if abj > row_size:
                    abj = row_size

                croped_Scale = frame[arr:abj, izq:dcha]  # Diferrents size are utilized
                image = cv2.resize(croped_Scale, (128, 128))

                # cv2.namedWindow('la', flags = cv2.WINDOW_NORMAL)
                # cv2.imshow('la', image)
                # cv2.waitKey()

                aux_vect = np.ravel(image)
                aux_X.append(aux_vect)

    return aux_X




def analize_results(y_real, y_pred, y_probabilidad, path):

    # La clase X ha sido mal clasificada
    mal_0 = 0
    mal_1 = 0
    mal_2 = 0
    mal_3 = 0


    # To know how samples of each data are bad classified.
    for pos, value in enumerate(y_pred):

        aux_pred = y_pred[pos]
        aux_real = y_real[pos]
        if aux_pred != aux_real and aux_real == 0:
            mal_0 += 1

        if aux_pred != aux_real and aux_real == 1:
            mal_1 += 1

        if aux_pred != aux_real and aux_real == 2:
            mal_2 += 1

        if aux_pred != aux_real and aux_real == 3:
            mal_3 += 1


    print ('Class 0 has been misclassified  %i times' % mal_0)
    print ('Class 1 has been misclassified  %i times' % mal_1)
    print ('Class 2 has been misclassified  %i times' % mal_2)
    print ('Class 3 has been misclassified  %i times' % mal_3)

    # To know false positive rate False negative, true positive and true negative. To do that,  0 is class real and attacks are class 1.

    TP = 0 #real 0 and pred 0
    TN = 0 #real 1 pred 1
    FN = 0 #real 0 pred 1
    FP = 0 # real 1 pred 0

    for pos, value in enumerate(y_pred):
        if y_real[pos] == 0 and y_pred[pos] == 0:
            TP += 1
        if y_real[pos] == 0 and y_pred[pos] == 1:
            FN += 1
        if y_real[pos] == 1 and y_pred[pos] == 1:
            TN += 1
        if y_real[pos] == 1 and y_pred[pos] == 0:
            FP += 1

    print ('TP, TN, FP, FN')
    print (TP, TN, FP, FN)

    precision, recall, threshold = precision_recall_curve(y_real, y_probabilidad[:,0])

    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    lw = 2

    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(recall, precision, lw=lw, color='navy', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall')
    plt.legend(loc="lower left")
    plt.savefig(path+'Precision-Recall.png')

    fpr, tpr, thresholds = roc_curve(y_real, y_probabilidad[:,0])


    print('Area under the roc curve:')
    roc_auc = roc_auc_score(y_real, y_probabilidad[:,0])
    print(roc_auc)

    plt.clf()
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.plot(fpr, tpr, 'b',
             label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(path+'ROC.png')

