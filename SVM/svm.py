import numpy as np
from sklearn.svm import SVC
#import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
from sklearn import metrics

def load_csv(file_name):
    values = read_csv(file_name, header=None).to_numpy()
    return values

##def pintar(X,y,svm):
# #   neg = np.where(y==0)
#  #  pos = np.where(y==1)
#   # plt.figure()
#    
#    #x1_min,x1_max = X[:,0].min(), X[:,0].max()
#    #x2_min,x2_max = X[:,1].min(), X[:,1].max()
#    xx1,xx2= np.meshgrid(np.linspace(x1_min,x1_max),np.linspace(x2_min,x2_max))
#    Z = svm.predict(np.c_[xx1.ravel(), xx2.ravel()])
#    Z = Z.reshape(xx1.shape)
#    plt.scatter(X[pos,0],X[pos,1],marker ='+',c='k')
#    plt.scatter(X[neg,0],X[neg,1],marker ='o',c='y')
#    plt.contour(xx1,xx2,Z,[0.5],linewidths=1,colors='g')

def supportv(Xent,yent,Xval,yval):
    val = np.array([0.01,0.03,0.05,0.1,0.3,0.5,1,3,5,10,15,30,50,100,150,300])
    maxilin = 0
    Csollin = 0
    print("Linear")
    for i in range(0,val.shape[0]):
        print(val[i])
        svm = SVC( kernel='linear', C=val[i])
        svm.fit(Xent,yent)
        w = svm.predict(Xval)        
        t = (w==yval)
        p = (np.count_nonzero(t)/yval.shape[0])*100
        #text = 'C='+repr(val[i])+'.Porcentaje='+repr(p)
        if(p>maxilin):
            Csollin = val[i]
            maxilin = p
        #print(text)
    textlin = 'Mejor solucion lineal: C = '+ repr(Csollin)+ ' . % = ' +repr(maxilin)
    maxigaus = 0
    Csolgaus = 0
    sigmasolgaus= 0
    print("Gauss")
    for i in range(0,val.shape[0]):
        print(val[i])
        for j in range(0,val.shape[0]):
            svm = SVC( kernel='rbf', C=val[i], gamma = 1/(2*val[j]**2))
            svm.fit(Xent,yent)
            w = svm.predict(Xval)
            t = (w==yval)
            p = (np.count_nonzero(t)/yval.shape[0])*100
            #text = 'C='+repr(val[i])+',sigma='+repr(val[j])+' .Porcentaje='+repr(p)
            if(p>maxigaus):
              Csolgaus = val[i]
              sigmasolgaus = val[j]
              maxigaus = p
            #print(text)
    text = 'Mejor solucion gaussiana: C = '+ repr(Csolgaus)+', Sigma = '+repr(sigmasolgaus)+ ' . % = ' +repr(maxigaus)
    print(textlin)
    print(text)

def prueb(Xent,yent,Xpr,ypr):
    val = np.array([50, 70, 100, 200])
    for i in range(0,val.shape[0]):
        print("SVC")
        svm = SVC(kernel="linear", C=val[i]) 
        print("fit")
        svm = svm.fit(Xent, yent)
        print("predict")
        y_pred = svm.predict(Xpr)
        print("accuracy")
        print("[",val[i], "] Accuracy: ",metrics.accuracy_score(ypr, y_pred))

def prueb2(Xent,yent,Xpr,ypr):
    val = np.array([0.01,0.03,0.05,0.1,0.3,0.5,1,3,5,10,15,30,50,100,150,300])
    sigma = 0.1
    
    for i in range(0,val.shape[0]):
        print("SVC")
        svm = SVC(kernel='rbf', C=val[i], gamma=1/(2*sigma**2))
        print("fit")
        svm = svm.fit(Xent, yent)
        print("predict")
        y_pred = svm.predict(Xpr)
        print("accuracy")
        print("[",val[i], "] Accuracy: ",metrics.accuracy_score(ypr, y_pred))


print("Todo")


entrenamiento = load_csv('Entrenamiento2.csv')
validacion = load_csv('Validacion2.csv')
prueba = load_csv('Prueba2.csv')
data = load_csv('Entrenamiento_Validacion.csv')

X_ent = entrenamiento[:, 1:-1]
y_ent = entrenamiento[:, -1]

X = data[:, 1:-1]
y = data[:, -1]

X_pr = prueba[:, 1:-1]
y_pr = prueba[:, -1]

supportv(X_ent,y_ent,X_pr,y_pr)
