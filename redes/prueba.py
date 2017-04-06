import numpy as np
from matplotlib import pyplot as plt

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


probabilidades= [0.1, 0.4, 0.7, 0.9, 0.4, 0.5, 0.2]

clasificacion_predict= [0, 0, 1, 1, 0, 1, 0]
clasificacion_real = [0, 1, 1, 1, 0 , 0, 0]

clas_user = 1
clas_attack = 0

probabilidades_son_de_clase = 0

sum_APCER, sum_BPCER = calculate_sumatorios(clasificacion_real, clasificacion_predict, clas_user, clas_attack)
# N_PAIS es como los verdaderos negativos en las etiquetas reales
#N_BF es como los verdaderos positicos en las etiqueta reales
N_PAIS = len(np.where(np.array(clasificacion_real)==clas_attack))
N_BF = len(np.where(np.array(clasificacion_real)==clas_user))

# Se calculan
APCER = (1/N_PAIS)*sum_APCER
BPCER = sum_BPCER/N_BF

APCER_ROC = np.array([10])
BPCER_ROC = np.array([10]) # Para 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9

probabilidades_umbral = []

for pos in range(0, 10):
    umbral = pos/10
    for prob_i in probabilidades:
        if prob_i>umbral:
            probabilidades_umbral.append(probabilidades_son_de_clase)
        else:
            probabilidades_umbral.append(1-probabilidades_son_de_clase)

            APCER_ROC[pos], BPCER_ROC[pos] = calculate_sumatorios(clasificacion_real, probabilidades_son_de_clase, clas_user, clas_attack)


