from pandas.io.parsers import read_csv
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def load_csv(file_name):
    values = read_csv(file_name, header=None).to_numpy()
    return values

def sigmoid(x):
	## cuidado con x > 50 devuelve 1
	##
	s = 1 / (1 + np.exp(-x))
	return s

def h(x,th):
    return sigmoid(np.matmul(x,th))

def porcentaje(th,X,y):
    res = h(X,th.T)
    maximo = np.argmax(res, axis = 1)
    comp = (maximo==y[:,0])*1
    g= np.count_nonzero(comp)
    return (g/len(comp))*100

def costReg(theta, XX, Y, reg):
	m = Y.size
	h = sigmoid(XX.dot(theta)) 
	cost = ((- 1 / m) * (np.dot(Y, np.log(h)) + np.dot((1 - Y), np.log(1 - h + 1e-6)))) + ((reg / (2 * m)) * (np.sum(np.power(theta, 2))))
	return cost

def gradient(theta, XX, Y, reg):
    H = sigmoid(np.matmul(XX, theta))
    m=len(Y)
    grad = (1 / m) * np.matmul(XX.T, H - Y)
    
    tmp=np.r_[[0],theta[1:]]

    thetaAux = theta
    thetaAux[0] = 0

    result = grad+(reg*tmp/m) + (reg / m * thetaAux)
    return result

def costeGrad(th,X,y,lambd):
    n = X.shape[0]
    grad = (1/n)*(np.matmul((h(X,th)-y[:,0]).T,X))
    grad =grad.T
    reg =(lambd/n)*th
    reg[0]= 0
    grad = grad+reg
    a = np.matmul(-y.T,np.log(h(X,th)))
    b = (np.matmul((1-y).T,np.log(1-h(X,th))).T)
    coste= (1/n)*np.sum(a-b)
    reg = (lambd/(2*n))* np.sum(th**2)
    coste = coste+reg
    return coste,grad

def oneVsAll(X,y,num_etiquetas,lambd):
    Xaux = np.hstack((np.ones((X.shape[0],1)),X))
    entrenador = np.zeros((num_etiquetas,X.shape[1]+1))
    for i in range(0,num_etiquetas):
        #entrenador[i]= opt.fmin_tnc(costeGrad,entrenador[i],args=(Xaux,(y==i)*1,lambd))[0]
        entrenador[i] = opt.minimize(costReg,entrenador[i],args=(X,(y==i)*1,lambd), jac=gradient).x   
    return entrenador

def errorlambda(X,y,Xval,yval):
    lmdb= np.array([0.001,0.003,0.01,0.03,0.1,0.3,1,3,5,10,15,30,70,120])
    n = X.shape[0]
    num_etiquetas = 21
    m=len(lmdb)
    porcent= np.zeros(m)
    porcval = np.zeros(m)
    for i in range(0,m):
        th=oneVsAll(X,y,num_etiquetas,lmdb[i])
        Xaux = np.hstack((np.ones((n,1)),X))
        porcent[i] = porcentaje(th,Xaux,y)
        Xaux = np.hstack((np.ones((Xval.shape[0],1)),Xval))
        porcval[i] = porcentaje(th,Xaux,yval)
    plt.xlabel('lambda')
    plt.ylabel('Error')
    plt.plot(lmdb,porcent,label="Entrenamiento", c='r')
    plt.plot(lmdb,porcval,label="Validacion", c= 'g')
    plt.legend()  

def main():
	entrenamiento = load_csv('Entrenamiento2.csv')
	validacion = load_csv('Validacion2.csv')
	prueba = load_csv('Prueba2.csv')

	X_ent = entrenamiento[:, 1:-1]
	y_ent = entrenamiento[:, -1]

	X_val = validacion[:, 1:-1]
	y_val = validacion[:, -1]

	X_pr = prueba[:, 1:-1]
	y_pr = prueba[:, -1]

	for i in y_val:
		print(i)

	errorlambda(X_ent, y_ent, X_val, y_val)




main()
