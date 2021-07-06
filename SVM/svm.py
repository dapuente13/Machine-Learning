import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
from sklearn import metrics
import time
import seaborn as sns

def load_csv(file_name):
    values = read_csv(file_name, header=None).to_numpy()
    return values

def printerrorlinear(val, porcent):
    plt.xlabel('lambda')
    plt.ylabel('Acierto')
    plt.plot(val,porcent,label="Train Set: Entrenamiento. Test Set: Validación", c='r')
    plt.savefig('SVM_Ent_Val.png')

def printerrorgauss(M, mn, best, val):
    cmap = sns.cm.rocket_r
    plt.figure(figsize=(10, 10))
    sns.color_palette("flare", as_cmap=True)
    sns.heatmap(M, square=True, annot=True, vmin=mn, vmax=best, cmap=cmap, xticklabels=val, yticklabels=val)
    plt.show()

def linear(Xent,yent,Xpr,ypr):
    val = np.array([0.001,0.003,0.01,0.03,0.1,0.3,1,3,5,10,50,70,100,150,180,200])
    best = 0
    _C = 0
    porcent= np.zeros(val.shape[0])
    for i in range(0,val.shape[0]):
        svm = SVC(kernel="linear", C=val[i]) 
        svm = svm.fit(Xent, yent)
        y_pred = svm.predict(Xpr)
        score = metrics.accuracy_score(ypr, y_pred)
        porcent[i] = score
        print("[",val[i], "] Accuracy: ",score)
        if best < score:       
            best = score
            _C = val[i]

    print("Linear - Best accuracy [", _C ,"]: ",best)
    printerrorlinear(val, porcent)

def gauss(Xent,yent,Xpr,ypr):
    val = np.array([0.001,0.003,0.01,0.03,0.1,0.3,1,3,5,10,50,70,100,150,180,200])
    max_C = 0
    max_sigma = 0
    best = 0
    mn = 100

    M = np.empty((val.shape[0],val.shape[0]))
    for i in range(0,val.shape[0]):
        for j in range(0,val.shape[0]):  
            svm = SVC(kernel='rbf', C=val[i], gamma=1/(2*val[j]**2))
            svm = svm.fit(Xent, yent)
            y_pred = svm.predict(Xpr)
            score = metrics.accuracy_score(ypr, y_pred)
            score = score * 100
            score = round(score, 2)
            M[i][j] = score
            print("[",val[i],",",val[j], "] Accuracy: ",score)
            if best < score:       
                best = score
                max_C = val[i]
                max_sigma = val[j]
            if mn > score:
                mn = score

    print("Gauss - Best accuracy [", max_C ,",",max_sigma,"]: ",best)
    printerrorgauss(M, mn, best, val)


def acierto_linear(Xent,yent,Xpr,ypr):
    _C = 0.001
    svm = SVC(kernel="linear", C=_C) 
    svm = svm.fit(Xent, yent)
    y_pred = svm.predict(Xpr)
    print("Accuracy: ",metrics.accuracy_score(ypr, y_pred))

def acierto_gauss(Xent,yent,Xpr,ypr):
    sigma = 5
    _C = 0.1
    svm = SVC(kernel='rbf', C=_C, gamma=1/(2*sigma**2))
    svm = svm.fit(Xent, yent)
    y_pred = svm.predict(Xpr)
    score = metrics.accuracy_score(ypr, y_pred)
    score = score * 100
    score = round(score, 2)
    print("Accuracy: ",score)

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

    #linear(X_ent,y_ent,X_val,y_val)
    #gauss(X_ent,y_ent,X_val,y_val)

    #print("X_ent")
    #acierto_linear(X_ent,y_ent,X_ent,y_ent)
    #acierto_linear(X_ent,y_ent,X_pr,y_pr)
    #acierto_linear(X_ent,y_ent,X_val,y_val)

    #print("X_ent_val")
    #acierto_linear(X_ent_val,y_ent_val,X_ent,y_ent)
    #acierto_linear(X_ent_val,y_ent_val,X_pr,y_pr)
    #acierto_linear(X_ent_val,y_ent_val,X_val,y_val)

    #print("todo")
    #acierto_linear(X,y,X_ent,y_ent)
    #acierto_linear(X,y,X_pr,y_pr)
    #acierto_linear(X,y,X_val,y_val)

    print("X_ent")
    acierto_gauss(X_ent,y_ent,X_ent,y_ent)
    acierto_gauss(X_ent,y_ent,X_pr,y_pr)
    acierto_gauss(X_ent,y_ent,X_val,y_val)

    print("X_ent_val")
    acierto_gauss(X_ent_val,y_ent_val,X_ent,y_ent)
    acierto_gauss(X_ent_val,y_ent_val,X_pr,y_pr)
    acierto_gauss(X_ent_val,y_ent_val,X_val,y_val)

    print("todo")
    acierto_gauss(X,y,X_ent,y_ent)
    acierto_gauss(X,y,X_pr,y_pr)
    acierto_gauss(X,y,X_val,y_val)

main()