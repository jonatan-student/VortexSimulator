import numpy as np
import matplotlib.pyplot as plt

Vorticity_data=np.genfromtxt('Nstest/Vorticity_atTime-100.csv', dtype= float, delimiter= ' ')
Xmesh = np.genfromtxt('Nstest/Xpos_atTime-100.csv', dtype = float, delimiter=' ')
Ymesh = np.genfromtxt('Nstest/Ypos_atTime-100.csv', dtype = float, delimiter =' ')
dx = np.gradient(Xmesh[0,:]).mean()
dy = np.gradient(Ymesh[:,0]).mean()
dwdx = np.zeros((len(Xmesh[0,:]), len(Ymesh[0,:])))
dwdy = np.zeros((len(Xmesh[0,:]),len(Ymesh[:,0])))
crit_points = []

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
for j in range(len(Ymesh[0,:])):
    dwdy[j,:] = np.divide(np.gradient(Vorticity_data[:,j]), dy)

for i in range(len(Vorticity_data[0,:])):
    for j in range(len(Vorticity_data[:,0])):
        if signchange(dwdx[i,j], dwdx[i-1, j]) == True and signchange(dwdy[i,j], dwdy[i,j-1]) == True:
            crit_points.append((Ymesh[i,j], Xmesh[i,j]))






plt.contour(Xmesh, Ymesh, Vorticity_data, 200, cmap = 'turbo')
plt.plot([c[0]for c in crit_points], [c[1]for c in crit_points], 'o')
plt.savefig('test.png')