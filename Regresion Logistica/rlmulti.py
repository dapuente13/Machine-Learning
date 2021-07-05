from pandas.io.parsers import read_csv
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pandas as pd
import seaborn as sns

def load_csv(file_name):
    values = read_csv(file_name, header=None).to_numpy()
    return values

def sigmoide(x):
    return 1/(1+np.exp(-x))

def h(x,th):
    return sigmoide(np.matmul(x,th))

def costeGrad(th,X,y,lambd):
    n = X.shape[0]
    grad = (1/n)*(np.matmul((h(X,th)-y).T,X))
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
        entrenador[i]= opt.fmin_tnc(costeGrad,entrenador[i],args=(Xaux,(y==i)*1,lambd))[0]
        #entrenador[i] = opt.minimize(coste,entrenador[i],args=(X,(y==i)*1,reg), jac=gradiente).x   
    return entrenador

def porcentaje(th,X,y):
    res = h(X,th.T)
    maximo = np.argmax(res, axis = 1)
    comp = (maximo==y)*1
    g= np.count_nonzero(comp)
    return (g/len(comp))*100

def errorlambda(X,y,Xval,yval):
    lmdb= np.array([0.001,0.003,0.01,0.03,0.1,0.3,1,3,5,10,15,30,50,70,100,150,170,180,190,200,210,220,230,300])
    n = X.shape[0]
    num_etiquetas = 4
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
    plt.savefig('error_lambda.png')       
    plt.show() 

def main():
	entrenamiento = load_csv('Entrenamiento2.csv')
	validacion = load_csv('Validacion2.csv')
	ent_val = load_csv('Ent_Val.csv')
	prueba = load_csv('Prueba2.csv')
	todo = load_csv('Train2.csv')
	rawdf =  pd.read_csv('Train3.csv').iloc[:,1:]

	rawdf.head()
	rawdf.describe()

	X = todo[:, 1:-1]
	y = todo[:, -1]

	X_ent = entrenamiento[:, 1:-1]
	y_ent = entrenamiento[:, -1]

	X_val = validacion[:, 1:-1]
	y_val = validacion[:, -1]

	X_ent_val = ent_val[:, 1:-1]
	y_ent_val = ent_val[:, -1]

	X_pr = prueba[:, 1:-1]
	y_pr = prueba[:, -1]

	rawdf.hist(figsize=(10, 10))
	plt.tight_layout()
	#plt.show()

	plt.figure(figsize=(10, 10))
	tempdf = rawdf.corr()[['Segmentation']].sort_values('Segmentation', ascending=False)
	sns.heatmap(tempdf, annot=True, vmin=-1, vmax=1)
	plt.show()

	#errorlambda(X_ent, y_ent, X_val, y_val)

	th = oneVsAll(X_ent_val,y_ent_val,4,100)

	Xp = np.concatenate((np.atleast_2d(np.ones(X_ent.shape[0])).T,X_ent),axis=1)
	p=porcentaje(th,Xp,y_ent)

	X_p_val = np.concatenate((np.atleast_2d(np.ones(X_val.shape[0])).T,X_val),axis=1)
	pval=porcentaje(th,X_p_val,y_val)

	X_p_prueba = np.concatenate((np.atleast_2d(np.ones(X_pr.shape[0])).T,X_pr),axis=1)
	ptest=porcentaje(th,X_p_prueba,y_pr)

	print("Porcentaje entrenamiento")
	print(p)
	print("Porcentaje validacion")
	print(pval)
	print("Porcentaje test")
	print(ptest)

main()