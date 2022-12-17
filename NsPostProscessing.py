import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans

_Directory = "NS00dalet"

if not os.path.exists(_Directory):
    os.mkdir(_Directory)

list_of_allPeaks = []

for Time in range(101):
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
    elif len(Peaks) <4:
        newPeaks = Peaks

    plt.contour(Xmesh, Ymesh, Vorticity_data, 50, cmap = 'turbo')
    #for c in newPeaks:
        #plt.plot(c[0], c[1], 'o', color = 'red')
    #plt.plot([c[0]for c in Saddles], [c[1]for c in Saddles], 'o', color = 'green')
    plt.savefig(f'{_Directory}/testat{Time}.png')
    list_of_allPeaks.append(newPeaks)

fig, ax = plt.subplots()
for peaks in list_of_allPeaks:
    for c in peaks:
        ax.plot(c[0], c[1], '.', color = 'blue')
fig.savefig(f'{_Directory}/testns.png')