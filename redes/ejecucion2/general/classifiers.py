from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from numpy import mean




def SVMClas(X_train, y_train, X_test, y_test, kernel):
    ########## SVM ###########

    print('----------SVM results ---------  KERNEL = ', kernel)
    svm_scores = []
    C_range = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 3, 5, 10]

    for Ci in C_range:
        SVMcla = SVC(C=Ci, kernel = kernel)
        scores = cross_val_score(SVMcla, X_train, y_train, cv=5,
                                 scoring='accuracy')  # Con neg_log_loss el predict tiene que ser con probabilidad
        svm_scores.append(scores.mean())

    optimaC = C_range[svm_scores.index(max(svm_scores))]
    print('optimaC', optimaC)

    SVM2 = SVC(C=optimaC, kernel='rbf', probability=True)
    SVM2.fit(X=X_train, y=y_train)

    SVM_pred = SVM2.predict(X_test)
    SVM_pred_prob = SVM2.predict_proba(X_test)
    scores_SVM = SVM2.score(X_test, y_test)

    return SVM_pred, SVM_pred_prob, scores_SVM


def KNNClas(X_train, y_train, X_test, y_test):

    ############  KNN  ##############

    print('---------- KNN results ---------')
    k_scores = []
    k_range = range(2, 32, 2)
    for i in k_range:
        neigh = neighbors.KNeighborsClassifier(n_neighbors=i)
        scores = cross_val_score(neigh, X_train, y_train, cv=5,
                                 scoring='accuracy')  # Con neg_log_loss el predict tiene que ser con probabilidad
        k_scores.append(mean(scores))

    optim_position = k_scores.index(max(k_scores))  # esto te da el indice
    optimk = k_range[optim_position]
    print('optim k value:', optimk)

    neigh = neighbors.KNeighborsClassifier(n_neighbors=(optimk))
    neigh.fit(X_train, y_train)

    knn_pred_prob = neigh.predict_proba(X_test)
    knn_pred = neigh.predict(X_test)
    scores_knn = neigh.score(X_test, y_test)

    return knn_pred, knn_pred_prob, scores_knn


def PCAClas(X_train, y_train, X_test, y_test):

    ############## PCA ################

    print('---------- PCA results ---------')

    ncomponentes = range(1, 2000, 10)
    PCA_scores = []
    for n in ncomponentes:
        pca = PCA(n_components=n)
        scores = cross_val_score(pca, X_train, y_train, cv=5,
                                 scoring='accuracy')  # Con neg_log_loss el predict tiene que ser con probabilidad
        print scores

        # PCA_scores.append(scores.mean())

    optimaNcomponentes = ncomponentes[PCA_scores.index(max(PCA_scores))]
    print('PCA optimous componentes number', optimaNcomponentes)

    pca = PCA(n_components=optimaNcomponentes)
    X_train_after_PCA = pca.fit_transform(X_train)
    X_test_after_PCA = pca.transform(X_test)

    return X_train_after_PCA, X_test_after_PCA



def DecisionTreeClas(X_train, y_train, X_test, y_test):
    ############ DECISION TREE ##############

    print('---------- DECISION TREES results ---------')

    depth_scores = []
    depth_range = range(2, 32, 2)
    for i in depth_range:
        dTree = DecisionTreeClassifier(max_depth=i)
        scores = cross_val_score(dTree, X_train, y_train, cv=5,
                                 scoring='accuracy')  # Con neg_log_loss el predict tiene que ser con probabilidad
        depth_scores.append(scores.mean())

    optim_position = depth_scores.index(max(depth_scores))  # esto te da el indice
    optim_depth = depth_range[optim_position]

    print('optim depth:', optim_depth)

    dTree = DecisionTreeClassifier(max_depth=optim_depth)
    dTree.fit(X_train, y_train)

    tree_prob = dTree.predict_proba(X_test)
    tree_pred = dTree.predict(X_test)
    scores_tree = dTree.score(X_test, y_test)

    return tree_pred, tree_prob, scores_tree

