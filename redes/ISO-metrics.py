import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def DETCurve(fps,fns):
    """
    Given false positive and false negative rates, produce a DET Curve.
    The false positive rate is assumed to be increasing while the false
    negative rate is assumed to be decreasing.
    """
    axis_min = min(fps[0],fns[-1])
    fig,ax = plt.subplots()
    plot(fps,fns)
    yscale('log')
    xscale('log')
    ticks_to_use = [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50]
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(ticks_to_use)
    ax.set_yticks(ticks_to_use)
    axis([0.001,50,0.001,50])

def calculate_sumatorios(clasificacion_real, clasificacion_predict, clas_user, clas_attack):
    sum_APCER = 0
    sum_BPCER = 0

    for pos in range(0, len(clasificacion_real)):
        if clasificacion_real[pos] == clas_attack and clasificacion_predict[pos] == clas_user:  # Para APCER tine que ser los ataques
            sum_APCER += 1

        if clasificacion_real[pos] == clas_user and clasificacion_predict[pos] == clas_attack:  # Para BPCER tine que ser los usuarios (bona fides)
            sum_BPCER += 1
    return sum_APCER, sum_BPCER


def metric(probabilidades, clasificacion_predict, clasificacion_real, probabilidades_son_de_clase, out_path):

    clas_user = 1
    clas_attack = 0


    sum_APCER, sum_BPCER = calculate_sumatorios(clasificacion_real, clasificacion_predict, clas_user, clas_attack)
    # N_PAIS es como los verdaderos negativos en las etiquetas reales
    #N_BF es como los verdaderos positicos en las etiqueta reales
    a = np.where(np.array(clasificacion_real)==clas_attack)
    N_PAIS = float(len(a[0]))

    b = np.where(np.array(clasificacion_real)==clas_user)
    N_BF = float(len(b[0]))
    # Se calculan
    APCER = float(float(1/N_PAIS)*sum_APCER)
    BPCER = float(sum_BPCER/N_BF)


    print 'APCER = ',APCER, 'BPCER = ', BPCER

    APCER_ROC = np.zeros(10)
    BPCER_ROC = np.zeros(10) # Para 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9

    for pos in range(0, 10):
        probabilidades_umbral = []
        umbral = pos*0.1
        for prob_i in probabilidades:
            if prob_i>umbral:
                probabilidades_umbral.append(probabilidades_son_de_clase)
            else:
                probabilidades_umbral.append(1-probabilidades_son_de_clase)

        aux_apcer, aux_bpcer = calculate_sumatorios(clasificacion_real, probabilidades_umbral, clas_user, clas_attack)
        APCER_ROC[pos] = float(float(aux_apcer)/N_PAIS)
        BPCER_ROC[pos] = float(float(aux_bpcer)/N_BF)


    plt.figure()
    plt.xlabel('APCER')
    plt.ylabel('BPCER')
    plt.title('APCER - BPCER curve')
    plt.plot(APCER_ROC,BPCER_ROC)
    plt.savefig(out_path+'APCER-BPCER curve.png')