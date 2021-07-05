from checkNNGradients import checkNNGradients
from displayData import displayData
from displayData import displayImage
from pandas.io.parsers import read_csv
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

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

def errorlmdb(X,y,Xval,yval,Theta1_ini,Theta2_ini):
	lmdb= np.array([0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,15,20,30,50,80,100,150,300])
	num_etiquetas=10
	num_entradas=32
	num_ocultas=25

	aux = np.reshape(Theta1_ini,(num_entradas+1)*num_ocultas)
	aux2 = np.reshape(Theta2_ini,(num_ocultas+1)*num_etiquetas)
	params_ini=np.concatenate((aux,aux2))
	porcentajeEnt = np.zeros(len(lmdb))
	porcentajeVal = np.zeros(len(lmdb))
	for i in range(0,len(lmdb)): #¿i == 1?
		res = opt.minimize(backprop,params_ini,args=(num_entradas,num_ocultas,num_etiquetas,X,y,lmdb[i]),jac=True)
	grad = res.jac

	Theta1 = np.reshape(grad[:num_ocultas*(num_entradas+1)],(num_ocultas, (num_entradas+1)))
	Theta2 = np.reshape(grad[num_ocultas*(num_entradas+1):],(num_etiquetas, (num_ocultas+1)))

	porcentajeEnt[i] = porcentajeRedNeuronal(Theta1, Theta2, X, y)
	porcentajeVal[i] = porcentajeRedNeuronal(Theta1, Theta2, Xval, yval)

	plt.xlabel('lambda')
	plt.ylabel('ac')
	plt.plot(lmdb,porcentajeEnt,label="Entrenamiento", c='r')
	plt.plot(lmdb,porcentajeVal,label="Validacion", c= 'g')
	plt.legend()
	plt.savefig('error_lambda.png')       
	plt.show()

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

def main():
	reg = 20 #λ = 1

	primera_capa = 32
	segunda_capa_oculta = 25
	num_labels = 10

	entrenamiento = load_csv('Entrenamiento2.csv')
	validacion = load_csv('Validacion2.csv')
	prueba = load_csv('Prueba2.csv')

	X_ent = entrenamiento[:, 1:-1]
	y_ent = entrenamiento[:, -1]

	X_val = validacion[:, 1:-1]
	y_val = validacion[:, -1]

	X_pr = prueba[:, 1:-1]
	y_pr = prueba[:, -1]

	m = len(y_ent)
	input_size = X_ent.shape[1]

	y = (y_ent - 1)
	y_onehot = np.zeros((m, num_labels))

	for i in range(m):
		y_onehot[i][y[i]] = 1

	#Unimos los datos de entrenamiento y validacion
	Xent_val=np.concatenate((X_ent,X_val))
	Yent_val=np.concatenate((y_ent,y_val))

	Theta1 = pesosAleatorios(primera_capa,segunda_capa_oculta)
	Theta2 = pesosAleatorios(segunda_capa_oculta,num_labels)
	Thetas = [Theta1, Theta2]

	unrolled_Thetas = [Thetas[i].ravel() for i,_ in enumerate(Thetas)]
	params = np.concatenate(unrolled_Thetas)

	#Entrenando a la red con 70 iteraciones y un valor de λ = 1
	optTheta = opt.minimize(fun=backprop, x0=params, args=(primera_capa, segunda_capa_oculta, num_labels,Xent_val, Yent_val, reg), method='TNC', jac=True, options={'maxiter': 70})
		
	Theta1_opt = np.reshape(optTheta.x[:segunda_capa_oculta * (primera_capa + 1)],(segunda_capa_oculta, (primera_capa + 1)))
	Theta2_opt = np.reshape(optTheta.x[segunda_capa_oculta * (primera_capa + 1): ], (num_labels, (segunda_capa_oculta + 1)))

	A1, A2, H = forward_propagate(X, Theta1_opt, Theta2_opt)

	print("Diferencia de gradiantes: ", str(np.sum(checkNNGradients(backprop, 1))))
	print(str(numAciertos(y,H)) + "% de precision\n")



main()

