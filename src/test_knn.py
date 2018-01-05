from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import numpy as np
import warnings
import random

totalX = []
totalY = []

for i in range(100):
    f = open("base/ModeA/File"+str(i)+".txt", "r")
    data = f.read().split("\t")
    data[20000] = 0
    totalX.append(data[:-1])
    totalY.append(0)

for i in range(100):
    f = open("base/ModeB/File"+str(i)+".txt", "r")
    data = f.read().split("\t")
    data[20000] = 0
    totalX.append(data[:-1])
    totalY.append(0)

for i in range(100):
    f = open("base/ModeC/File"+str(i)+".txt", "r")
    data = f.read().split("\t")
    data[20000] = 0
    totalX.append(data[:-1])
    totalY.append(0)

for i in range(100):
    f = open("base/ModeD/File"+str(i)+".txt", "r")
    data = f.read().split("\t")
    if(len(data) <= 20000):
        continue
    data[20000] = 0
    totalX.append(data[:-1])
    totalY.append(0)
'''
fftX = np.fft.fft(totalX)
pca = PCA(n_components=99, svd_solver='full')
warnings.filterwarnings("ignore")
newX = pca.fit_transform(fftX)
'''
newX = np.fft.fft(totalX)
#newX = totalX
clf = NearestNeighbors(n_neighbors=3)
distances, indices = clf.fit(newX).kneighbors(newX)

alpha = [0]*len(distances)
for i in range(len(distances)):
    alpha[i] = sum(distances[i])
print (alpha)

testX = []
testY = []

for i in range(100):
    f = open("base/ModeM/File" + str(i) + ".txt", "r")
    data = f.read().split("\t")
    if (len(data) <= 20000):
        continue
    data[20000] = 1
    testX.append(data[:-1])
    testY.append(1)
'''
fftTestX = np.fft.fft(testX)
pca = PCA(n_components=99, svd_solver='full')
newTestX = pca.fit_transform(fftTestX)
'''
newTestX = np.fft.fft(testX)
#newTestX = testX

f = open("results.txt", "w")
clf = NearestNeighbors(n_neighbors=3)
distances, indices = clf.fit(newX).kneighbors(newTestX)

conf = 0.005
for k in range(20):
    num = 0.0
    for i in range(len(newTestX)):
        b = 0.0
        strangeness_i = sum(distances[i])
        #print (strangeness_i)
        for j in range(len(alpha)):
            if strangeness_i <= alpha[j]:
                b += 1.0
        pvalue = (b+1.0)/(float(len(alpha))+1.0)
        #print (pvalue)
        if pvalue < conf:
            num += 1.0
    print ("Accuracy for confidence "+ str(1.0-conf) +" : " + str(num/float(len(testY))*100) + "%")
    conf += 0.005
