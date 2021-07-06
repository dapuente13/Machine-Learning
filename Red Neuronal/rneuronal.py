from checkNNGradients import checkNNGradients
from displayData import displayData
from displayData import displayImage
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.io import loadmat
import numpy as np
from pandas.io.parsers import read_csv

def load_csv(file_name):
    values = read_csv(file_name, header=None).to_numpy()
    return values

def sigmoide(x):
    return 1 / (1 + np.exp(-x))

def derivada_sigmoide(x):
    return (sigmoide(x) * (1.0 - sigmoide(x)))

def forward_propagate(X, Theta1, Theta2):
    m = X.shape[0]
    A1 = np.hstack([np.ones([m, 1]), X])
    Z2 = np.dot(A1, Theta1.T)
    A2 = np.hstack([np.ones([m, 1]), sigmoide(Z2)])
    Z3 = np.dot(A2, Theta2.T)
    H = sigmoide(Z3)
    return A1, A2, H

def funcion_cost(m, h, y):
    J = 0
    for i in range(m):
        J += np.sum(-y[i] * np.log(h[i]) - (1 - y[i]) * np.log(1 - h[i]))
    return (J / m)

def costReg(m, h, Y, reg, theta1, theta2):
    return funcion_cost(m, h, Y) + ((reg / (2 * m)) * ((np.sum(np.square(theta1[:, 1:]))) + (np.sum(np.square(theta2[:,1:])))))    

    
def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
    # backprop devuelve una tupla (coste , gradiente) con el coste y el gradiente de
    # una red neuronal de tres capas , con num_entradas , num_ocultas nodos en la capa
    # oculta y num_etiquetas nodos en l a capa de salida . Si m es el número de ejemplos
    # de entrenamiento , la dimensión de ’X’ es (m, num_entradas) y la de ’y’ es
    # (m, num_etiquetas)

    m = X.shape[0]
    
    Theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)],(num_ocultas, (num_entradas + 1)))

    Theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1): ],(num_etiquetas, (num_ocultas + 1)))
    
    A1, A2, H = forward_propagate(X, Theta1, Theta2)
    
    Delta1 = np.zeros_like(Theta1)
    Delta2 = np.zeros_like(Theta2)

    for t in range(m):
        a1t = A1[t, :] # (401,)
        a2t = A2[t, :] # (26,)
        ht = H[t, :] # (10,)
        yt = y[t] # (10,)
        d3t = ht - yt # (10,)
        d2t = np.dot(Theta2.T, d3t) * (a2t * (1 - a2t)) # (26,)
        Delta1 = Delta1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
        Delta2 = Delta2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])
        
    Delta1 = Delta1 / m
    Delta2 = Delta2 / m
    
    Delta1[:, 1:] = Delta1[:, 1:] + (reg * Theta1[:, 1:]) / m
    Delta2[:, 1:] = Delta2[:, 1:] + (reg * Theta2[:, 1:]) / m
    
    coste = costReg(m, H, y, reg, Theta1, Theta2)
    
    gradiente = np.concatenate((np.ravel(Delta1), np.ravel(Delta2)))
    
    return coste, gradiente

def pesosAleatorios(L_in, L_out):
    ini = 0.12
    pesos = np.random.rand((L_in+1)*L_out)*(2*ini) - ini
    pesos = np.reshape(pesos, (L_out,1+L_in))
    return pesos

def numAciertos(Y, h):
    aciertos = 0
    totales = len(Y)
    dimThetas = len(h)

    for i in range(dimThetas):
        r = np.argmax(h[i])
        if(r == Y[i]):
            aciertos = aciertos + 1     

    porcentaje = aciertos / totales * 100
    return porcentaje

def printerrorlambda(primera_capa, num_labels, X_ent, y_ent, X_pr, y_pr):
    reg =np.array([3])
    best = 0
    lamda = 3
    segunda_capa_oculta = 10
    porcent= np.zeros(reg.shape[0])
    for i in range(0,reg.shape[0]):
        Theta1 = pesosAleatorios(primera_capa, segunda_capa_oculta)
        Theta2 = pesosAleatorios(segunda_capa_oculta, num_labels)

        Thetas = [Theta1, Theta2]

        unrolled_Thetas = [Thetas[i].ravel() for i,_ in enumerate(Thetas)]
        params = np.concatenate(unrolled_Thetas)

        optTheta = opt.minimize(fun=backprop, x0=params, args=(primera_capa, segunda_capa_oculta, num_labels,X_ent,y_ent,reg[i]), method='TNC', jac=True, options={'maxiter': 70})

        Theta1_opt = np.reshape(optTheta.x[:segunda_capa_oculta * (primera_capa + 1)],
        (segunda_capa_oculta, (primera_capa + 1)))
        Theta2_opt = np.reshape(optTheta.x[segunda_capa_oculta * (primera_capa + 1): ], 
        (num_labels, (segunda_capa_oculta + 1)))
        
        A1, A2, H = forward_propagate(X_pr, Theta1_opt, Theta2_opt)
        score = numAciertos(y_pr,H)
        porcent[i] = score
        print(str(score) + "% de precision ", segunda_capa_oculta , " ", reg[i])
        if best < score:
            best = score
            lamda = reg[i]
        

    #print("Best accuracy [", lamda   ,"]: ",best)

    #plt.xlabel('lambda')
    #plt.ylabel('Acierto')
    #plt.plot(reg,porcent,label="Test Set: Entrenamiento", c='g')
    #plt.savefig('Rneuro_100_5.png')
    #plt.show()
    return best

def main():
    entrenamiento = load_csv('Entrenamiento2.csv')
    validacion = load_csv('Validacion2.csv')
    prueba = load_csv('Prueba2.csv')
    data = load_csv('Ent_Val.csv')
    todo = load_csv('Train2.csv')

    X = todo[:, 1:-1]
    y = todo[:, -1]

    X_ent = entrenamiento[:, 1:-1]
    y_ent = entrenamiento[:, -1]

    X_val = validacion[:, 1:-1]
    y_val = validacion[:, -1]

    X_ent_val = data[:, 1:-1]
    y_ent_val = data[:, -1]

    X_pr = prueba[:, 1:-1]
    y_pr = prueba[:, -1]

    primera_capa = X_ent.shape[1] #400
    num_labels = 4



    m = len(y_ent)
    y_ent = (y_ent - 1)
    y_onehot = np.zeros((m, num_labels))  # 5000 x 10


    #printerrorlambda(primera_capa, num_labels, X_ent, y_ent, X_val, y_val)

    print("X_ent")
    printerrorlambda(primera_capa, num_labels, X_ent, y_ent, X_ent, y_ent)
    printerrorlambda(primera_capa, num_labels, X_ent, y_ent, X_val, y_val)
    printerrorlambda(primera_capa, num_labels, X_ent, y_ent, X_pr, y_pr)

    print("X_ent_val")
    printerrorlambda(primera_capa, num_labels, X_ent_val, y_ent_val, X_ent, y_ent)
    printerrorlambda(primera_capa, num_labels, X_ent_val, y_ent_val, X_val, y_val)
    printerrorlambda(primera_capa, num_labels, X_ent_val, y_ent_val, X_pr, y_pr)

    print("todo")
    printerrorlambda(primera_capa, num_labels, X, y, X_ent, y_ent)
    printerrorlambda(primera_capa, num_labels, X, y, X_val, y_val)
    printerrorlambda(primera_capa, num_labels, X, y, X_pr, y_pr)

main()