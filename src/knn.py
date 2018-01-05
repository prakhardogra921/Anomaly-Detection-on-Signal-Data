from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import numpy as np

totalX = []

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
    if(len(data) <= 20000):
        continue
    data[20000] = 0
    totalX.append(data[:-1])

#applying FFT and PCA on loaded data
fftX = np.fft.fft(totalX)
pca = PCA(svd_solver='full')
newX = pca.fit_transform(fftX)

clf = NearestNeighbors(n_neighbors=4)               #taking 4 neighbours since the each point will also consider itself as nearest neighbor
distances, indices = clf.fit(newX).kneighbors(newX)
alpha = [0]*len(distances)

#Calculating KNN scores from KNN distances
for i in range(len(distances)):
    alpha[i] = sum(distances[i])

alpha.sort(reverse=True)

testX = []
for i in range(499):                                #loads Mode M data as test data
    f = open("Test/TestWT/Data"+str(i+1)+".txt", "r")
    data = f.read().split("\t")
    testX.append(data[:-1])

#applying FFT and PCA on loaded data
fftTestX = np.fft.fft(testX)
pca = PCA(n_components=399, svd_solver='full')
newTestX = pca.fit_transform(fftTestX)


f = open("results.txt", "w")                        #writing results to text file

#Applying KNN on new data set
clf = NearestNeighbors(n_neighbors=3)
distances, indices = clf.fit(newX).kneighbors(newTestX)

#StrOUD Algorithm with KNN as strangeness function
for i in range(len(newTestX)):
    b = 0.0
    strangeness_i = sum(distances[i])
    print (strangeness_i)
    for j in range(len(alpha)):
        if strangeness_i > alpha[j]:
            break
        b += 1.0
    pvalue = (b+1.0)/(float(len(newTestX))+1.0)
    f.write(str(pvalue)+"\n")
