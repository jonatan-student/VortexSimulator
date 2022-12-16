import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans

_Directory = "NS01dalet"

if not os.path.exists(_Directory):
    os.mkdir(_Directory)

T = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]#, 5.5, 6, 6.5, 7, 7.5, 8, 8.5,9,9.5,10 ,10.5, 11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5,18,18.5,19,19.5,20,20.5,21,21.5,22,22.5,23,23.5,24,24.5,25,25.5,26,26.5,27,27.5,28,28.5,29,29.5,30,30.5,31,31.5,32,32.5,33,33.5,34,34.5,35,35.5,36,36.5,37,37.5,38,38.5,39,39.5,40,40.5,41,41.5,42,42.5,43,43.5,44,44.5,45,45.5,46,46.5,47,47.5,48,48.5,49,49.5,50]
list_of_allPeaks = []

for Time in range(51):
    print(Time)
    Vorticity_data=np.genfromtxt(f'Nstest/Vorticity_atTime-{Time}.csv', dtype= float, delimiter= ' ')
    Xmesh = np.genfromtxt(f'Nstest/Xpos_atTime-0.csv', dtype = float, delimiter=' ')
    Ymesh = np.genfromtxt(f'Nstest/Ypos_atTime-0.csv', dtype = float, delimiter =' ')
    dx = np.gradient(Xmesh[0,:]).mean()
    dy = np.gradient(Ymesh[:,0]).mean()
    dwdx = np.zeros((len(Xmesh[0,:]), len(Ymesh[0,:])))
    dwdy = np.zeros((len(Xmesh[0,:]),len(Ymesh[:,0])))
    dwdx2 = np.zeros((len(Xmesh[0,:]), len(Ymesh[0,:])))
    dwdy2 = np.zeros((len(Xmesh[0,:]), len(Ymesh[:,0])))
    dwdxy =dwdy2
    detH = dwdy2
    Peaks = []
    Saddles = []

    def signchange(focus, last):
        changed = False
        if focus*last > 0:
            changed = False
        if focus*last < 0:
            changed = True
        if focus*last == 0:
            changed = False
        return(changed)

    for i in range(len(Xmesh[0,:])):
        dwdx[:,i] = np.divide(np.gradient(Vorticity_data[i,:]), dx)
        dwdx2[:,i] = np.divide(np.gradient(dwdx[:,i]), dx)
        dwdxy[:,i] = np.divide(np.gradient(dwdxy[:,i]), dy)
    for j in range(len(Ymesh[0,:])):
        dwdy[j,:] = np.divide(np.gradient(Vorticity_data[:,j]), dy)
        dwdy2[j,:] = np.divide(np.gradient(dwdy[:,j]), dy)

    for i in range(len(Vorticity_data[0,:])):
        for j in range(len(Vorticity_data[:,0])):
            detH[i,j] = (dwdy2[i,j]*dwdx2[i,j])-(dwdxy[i,j]**2)
            #print((detH[i,j], Ymesh[i,j], Xmesh[i,j]))
            if signchange(dwdx[i,j], dwdx[i-1, j]) == True and signchange(dwdy[i,j], dwdy[i,j-1]) == True:
                    Peaks.append((Ymesh[i,j], Xmesh[i,j]))
    
    if len(Peaks)>=4:
        kmeans = KMeans(4, n_init='auto')
        kmeans.fit(Peaks)
        newPeaks = kmeans.cluster_centers_
        print([(c[0], c[1]) for c in newPeaks])




    plt.contour(Xmesh, Ymesh, Vorticity_data, 200, cmap = 'turbo')
    for c in newPeaks:
        plt.plot(c[0], c[1], 'o', color = 'red')
    #plt.plot([c[0]for c in Saddles], [c[1]for c in Saddles], 'o', color = 'green')
    plt.savefig(f'{_Directory}/testat{Time}.png')
    list_of_allPeaks.append(newPeaks)

fig, ax = plt.subplots()
for peaks in list_of_allPeaks:
    for c in peaks:
        ax.plot(c[0], c[1], '.', color = 'blue')
fig.savefig(f'{_Directory}/testns.png')