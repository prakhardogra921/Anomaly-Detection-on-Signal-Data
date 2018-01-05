from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import numpy as np
import warnings

totalX = []

#reachDist() calculates the reach distance of each point to MinPts around it
def reachDist(df, MinPts, knnDist):
    clf = NearestNeighbors(n_neighbors=MinPts)
    clf.fit(df)
    distancesMinPts, indicesMinPts = clf.kneighbors(df)
    distancesMinPts[:,0] = np.amax(distancesMinPts,axis=1)
    distancesMinPts[:,1] = np.amax(distancesMinPts,axis=1)
    distancesMinPts[:,2] = np.amax(distancesMinPts,axis=1)
    return distancesMinPts, indicesMinPts

#lrd calculates the Local Reachability Density
def lrd(MinPts,knnDistMinPts):
    return (MinPts/np.sum(knnDistMinPts,axis=1))

#Finally lof calculates LOF outlier scores
def lof(Ird,MinPts,dsts):
    lof=[]
    for item in dsts:
       tempIrd = np.divide(Ird[item[1:]],Ird[item[0]])
       lof.append(tempIrd.sum()/MinPts)
    return lof

for i in range(100):                                #loads Mode A data
    f = open("base/ModeA/File"+str(i)+".txt", "r")
    data = f.read().split("\t")
    data[20000] = 0
    totalX.append(data[:-1])

for i in range(100):                                #loads Mode B data
    f = open("base/ModeB/File"+str(i)+".txt", "r")
    data = f.read().split("\t")
    data[20000] = 0
    totalX.append(data[:-1])

for i in range(100):                                #loads Mode C data
    f = open("base/ModeC/File"+str(i)+".txt", "r")
    data = f.read().split("\t")
    data[20000] = 0
    totalX.append(data[:-1])

for i in range(100):                                #loads Mode D data
    f = open("base/ModeD/File"+str(i)+".txt", "r")
    data = f.read().split("\t")
    if(len(data) <= 20000):                         #there was a file that was empty in the training folder
        continue
    data[20000] = 0
    totalX.append(data[:-1])

#in case we want to apply FFT and PCA
'''
fftX = np.fft.fft(totalX)
pca = PCA(svd_solver='full')
warnings.filterwarnings("ignore")
newX = pca.fit_transform(fftX)
'''
#newX = np.fft.fft(totalX)                          #in case we want to apply only FFT
newX = totalX

#calculating KNN for each point
clf = NearestNeighbors(n_neighbors=3)
distances, indices = clf.fit(newX).kneighbors(newX)

#calculating LOF scores for each point
m = 10
reachdist, reachindices = reachDist(newX, m, distances)
irdMatrix = lrd(m, reachdist)
lofScores = lof(irdMatrix, m, reachindices)

alpha = lofScores
alpha.sort()

testX = []
for i in range(499):                                #loads Mode M data as test data
    f = open("Test/TestWT/Data"+str(i+1)+".txt", "r")
    data = f.read().split("\t")
    testX.append(data[:-1])

'''
fftTestX = np.fft.fft(testX)
pca = PCA(n_components=399, svd_solver='full')
warnings.filterwarnings("ignore")
newTestX = pca.fit_transform(fftTestX)
'''
#newTestX = np.fft.fft(testX)                       #in case we want to apply only FFT
newTestX = testX

f = open("results.txt", "w")                        #writing results to text file
clf = NearestNeighbors(n_neighbors=3)
distances, indices = clf.fit(newTestX).kneighbors(newTestX)

#LOF score calculation
m = 10
reachdist, reachindices = reachDist(newTestX, m, distances)
irdMatrix = lrd(m, reachdist)
lofScores = lof(irdMatrix, m, reachindices)

#StrOUD Algorithm with LOF as strangeness function
for i in range(len(newTestX)):
    b = 0.0
    strangeness_i = lofScores[i]
    for j in range(len(alpha)):
        if strangeness_i <= alpha[j]:
            b += 1.0
    pvalue = (b+1.0)/(float(len(alpha))+1.0)
    f.write(str(pvalue)+"\n")
