import numpy as np
import pygame
import matplotlib.pyplot as plt
import os
import sys

#initialize pygame and plot generation
pygame.init()

if not os.path.exists("Figures"):
    os.mkdir("Figures")

#__________glbl variables____________
#display stuff
SCREEN_HEIGHT = 900
SCREEN_WIDTH = 900
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
FONT = pygame.font.Font(pygame.font.get_default_font(), 20)

#simulation stuff
ZeroZero =  SCREEN_WIDTH/2, SCREEN_HEIGHT/2  #<--recenter (0,0) coordinate in center of graph
dt =  .1                                     #<--timestep, lower means more accurate
Sim_type = 'Core Growth'                     #<--Choose whether to run point vortex or coregrowth simulation
Simulation = False                           #<--choose wether or not you want to see the simulation as it runs
Strength = -1, .5, .5                         #<--Choose initial vortex strengths here
SupremeCounter = 1e3                         #<--number of iterations until loop breaks
viscosity = 1e3                              #<--Choose a viscosity of the fluid here

#Point vortexs are defined here
class pvortex():
    def __init__(self, VortexStrength, X0, Y0):
        self.Vstrength = VortexStrength
        self.position = X0+1j*Y0
        self.pastpositions = []

    def updatePosition(self, VelocityVector):
        self.position += VelocityVector *dt
        self.pastpositions.append(self.position)
        if Simulation == True:
            self.draw((np.real(self.position), np.imag(self.position)))

    def move(self, Other1Strength, Other2Strength, Other1pos, Other2pos):
        conj_velVec = (1/(2*np.pi*1j)) * ((Other1Strength/(self.position-Other1pos)) + (Other2Strength/(self.position - Other2pos)))
        self.updatePosition(np.conj(conj_velVec))
        
    def draw(self, pos):
        pygame.draw.circle(SCREEN, (0,0,255), (pos[0], pos[1]), 4)
        for i in self.pastpositions:
            pygame.draw.circle(SCREEN, (255, 0, 0) , (np.real(i), np.imag(i)), 2)

class cgvortex():
    def __init__(self,VortexStr, x0, y0):
        self.Vstrength = VortexStr
        self.position = x0 + 1j*y0
        self.pastpositions = []

    def updatePosition(self, V_vector):
        self.position += V_vector*dt
        
    def movepeak(self, other1, other2, t):
        self.distribution1 = 1-np.exp(-1*(np.abs(self.position-other1.position)**2)/(4*viscosity*t))
        self.distribution2 = 1-np.exp(-1*(np.abs(self.position-other2.position)**2)/(4*viscosity*t))
        conjegate_velocity_vector = (1/(2*np.pi*1j)) * (((other1.Vstrength/(self.position-other1.position))*self.distribution1)+((other2.Vstrength/(self.position-other2.position))*self.distribution2))
        self.updatePosition(np.conj(conjegate_velocity_vector))

    def drawpeak(self, pos):
        pygame.draw.circle(SCREEN, (0,0,255), (pos[1], pos[2]))

#interaction between vortexs is partly handled here
class Vortex_interaction():
    def __init__(self):
        self.SaveD = []
        self.fieldPos = []
        self.critical_points = []
        self.Vort_peak = []
        self.Saddles = []

    def updatePVortexes(self, v1, v2, v3):
        v1_pos_temp = v1.position
        v2_pos_temp = v2.position
        v3_pos_temp = v3.position
        v1_vstrength_temp = v1.Vstrength
        v2_vstrength_temp = v2.Vstrength
        v3_vstrength_temp = v3.Vstrength
        v1.move(v2_vstrength_temp,v3_vstrength_temp,v2_pos_temp,v3_pos_temp)
        v2.move(v1_vstrength_temp,v3_vstrength_temp,v1_pos_temp,v3_pos_temp)
        v3.move(v1_vstrength_temp, v2_vstrength_temp,v1_pos_temp,v2_pos_temp)

    def updateCGvortexes(self, v1, v2, v3, t):
        self.V1 = v1
        self.V2 = v2
        self.V3 = v3
        v1.movepeak(self.V2, self.V3,t)
        v2.movepeak(self.V1, self.V3,t)
        v3.movepeak(self.V1, self.V2,t)

    def vorticity_field(self, vortices, t):
        x = np.linspace(-9*SCREEN_WIDTH,10*SCREEN_WIDTH,  100)
        y = np.linspace(-9*SCREEN_HEIGHT, 10*SCREEN_HEIGHT, 100)
        Vort = np.zeros((len(x),len(y)))
        dwdx = np.zeros((len(x), len(y)))
        dwdy = np.zeros((len(x), len(y)))
        dwdx2 = np.zeros((len(x),len(y)))
        dwdy2 = np.zeros((len(x),len(y)))
        dwdxdy = np.zeros((len(x),len(y)))
        dx = np.gradient(x).mean()
        dy = np.gradient(y).mean()
        crit_points = []
        for i in range(len(x)):
            for j in range(len(y)):
                z = x[i]+1j*y[j]
                Vort[i,j] = ((vortices[0].Vstrength/(4*np.pi*viscosity*t))*np.exp(-1*(np.abs(z-vortices[0].position)**2)/(4*viscosity*t)))+ ((vortices[1].Vstrength/(4*np.pi*viscosity*t))*np.exp(-1*(np.abs(z-vortices[1].position)**2)/(4*viscosity*t)))+((vortices[2].Vstrength/(4*np.pi*viscosity*t))*np.exp(-1*(np.abs(z-vortices[2].position)**2)/(4*viscosity*t)))
        for j in range(len(y)):
            dwdx[:, j] = np.divide(np.gradient(Vort[:,j]), dx)
            dwdx2[:, j] = np.divide(np.gradient(dwdx[:,j]), dx)
            dwdxdy[:, j] = np.divide(np.gradient(dwdx[:,j]),dy)
        for i in range(len(x)):
            dwdy[i, :] = np.divide(np.gradient(Vort[i,:]), dy)
            dwdy2[i, :] = np.divide(np.gradient(dwdy[i,:]), dy)
        for i in range(len(x)):
            for j in range(len(y)):
                signx = dwdx[i,j]/np.abs(dwdx[i,j])
                nextsignx = dwdx[i-1,j]/np.abs(dwdx[i-1,j])
                signy = dwdy[i,j]/np.abs(dwdy[i,j])
                nextsigny = dwdy[i, j-1]/np.abs(dwdy[i,j-1])
                detH = dwdx2*dwdy2-dwdxdy**2
                if signx != nextsignx and signy !=nextsigny:
                    if dwdx[i,j] == 0:
                        xcoord = x[i]+1j*y[j]
                    if dwdx[i-1,j]== 0:
                        xcoord = x[i-1]+1j*y[j]
                    if dwdy[i,j]== 0:
                        ycoord = x[i]+1j*y[j]
                    if dwdy[i, j-1]==0:
                        ycoord = x[i]+1j*y[j-1]
                    else:
                        ax = (dwdx[i-1,j]-dwdx[i,j])/(x[i-1]-x[i])
                        ay = (dwdy[i,j-1]-dwdy[i,j])/(y[j-1]-y[j])
                        bx = dwdx[i,j] - (ax*x[i])
                        by = dwdy[i,j]- (ay*y[j])
                        xcoord = -(bx/ax)+1j*y[j]
                        ycoord = x[i]+1j*-(by/ay)
                    zed = (xcoord+ycoord)/2
                    crit_points.append((np.real(zed), np.imag(zed)))
        self.critical_points.append(crit_points)
        print(detH)


    def findStreamlines(self, v1, v2, v3, t):
        x = np.linspace(-9*SCREEN_WIDTH,10*SCREEN_WIDTH,  60)
        y = np.linspace(-9*SCREEN_HEIGHT, 10*SCREEN_HEIGHT, 60)
        X, Y = np.meshgrid(x,y)
        z = X+1j*Y

        #velocity field
        dist1 = 1-np.exp(-1*(np.abs(z-v1.position)**2)/(4*viscosity*t))
        dist2 = 1-np.exp(-1*(np.abs(z-v2.position)**2)/(4*viscosity*t))
        dist3 = 1-np.exp(-1*(np.abs(z-v3.position)**2)/(4*viscosity*t))
        conjugate_Zdot = (1/(2*np.pi*1j))*(((v1.Vstrength/(z-v1.position))*dist1)+((v2.Vstrength/(z-v2.position))*dist2)+((v3.Vstrength/(z-v3.position))*dist3))
        Zdot= np.conj(conjugate_Zdot)

        #vorticity field
        Vorticity = ((v1.Vstrength/(4*np.pi*viscosity*t))*np.exp(-1*(np.abs(z-v1.position)**2)/(4*viscosity*t)))+ ((v2.Vstrength/(4*np.pi*viscosity*t))*np.exp(-1*(np.abs(z-v2.position)**2)/(4*viscosity*t)))+((v3.Vstrength/(4*np.pi*viscosity*t))*np.exp(-1*(np.abs(z-v3.position)**2)/(4*viscosity*t)))
        self.SaveD.append((Zdot, t, Vorticity))
        self.fieldPos.append((x, y, X, Y))

######initializeing some parameters for le sim
run = True
t = 0
vortexs = Vortex_interaction()

###initialize Point vortexes if chosen for this simulation
if Sim_type == 'Point Vortex':
    V1 = pvortex(Strength[0], ZeroZero[0], ZeroZero[1]-100)
    V2 = pvortex(Strength[1], ZeroZero[0]-(200*np.tan(np.radians(30))), ZeroZero[1]+100)
    V3 = pvortex(Strength[2], ZeroZero[0]+(200*np.tan(np.radians(30))), ZeroZero[1]+100)


###initialize Gausian vortexes if chosen for this simulation
if Sim_type == 'Core Growth':
    V1 = cgvortex(Strength[0], ZeroZero[0], ZeroZero[1]-1500)
    V2 = cgvortex(Strength[1], ZeroZero[0]-(3000*np.tan(np.radians(30))), ZeroZero[1]+1500)
    V3 = cgvortex(Strength[2], ZeroZero[0]+(3000*np.tan(np.radians(30))), ZeroZero[1]+1500)

###SIMULATION LOOP IS HERE
while run == True:
    t = round(t + dt,1)
    #create white Background
    SCREEN.fill((255,255,255))
    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    if Sim_type== 'Point Vortex':
        vortexs.updatePVortexes(V1, V2, V3)

    if Sim_type == 'Core Growth':
        vortexs.updateCGvortexes(V1, V2, V3, t)
        if t%40 == 0:
            vortexs.findStreamlines(V1, V2, V3, t)
            vortexs.vorticity_field((V1,V2,V3), t)
    if Simulation == False:
        text = FONT.render(f'Supreme Counter = {SupremeCounter}', True, (0,0,0))
        SCREEN.blit(text, ZeroZero)
    SupremeCounter -=1
    if SupremeCounter <= 0:
        break
    print(t)
    pygame.display.update()


if Sim_type == 'Point Vortex':
    fig, ax = plt.subplots()
    ax.plot(np.real(V1.pastpositions), np.imag(V1.pastpositions), label = 'V1')
    ax.plot(np.real(V2.pastpositions), np.imag(V2.pastpositions), label = 'V2')
    ax.plot(np.real(V3.pastpositions), np.imag(V3.pastpositions), label = 'V3')

    ax.plot(np.real(V1.pastpositions[0]), np.imag(V1.pastpositions[0]), 'o', label = 'V1 Z0')
    ax.plot(np.real(V2.pastpositions[0]), np.imag(V2.pastpositions[0]), 'o', label = 'V2 Z0')
    ax.plot(np.real(V3.pastpositions[0]), np.imag(V3.pastpositions[0]), 'o', label = 'V3 Z0')

    ax.plot(np.real(V1.pastpositions[-1]), np.imag(V1.pastpositions[-1]), 'o', label = 'V1 Zfinal')
    ax.plot(np.real(V2.pastpositions[-1]), np.imag(V2.pastpositions[-1]), 'o', label = 'V2 Zfinal')
    ax.plot(np.real(V3.pastpositions[-1]), np.imag(V3.pastpositions[-1]), 'o', label = 'V3 Zfinal')

    ax.plot([np.real(V1.pastpositions[0]), np.real(V2.pastpositions[0])], [np.imag(V1.pastpositions[0]), np.imag(V2.pastpositions[0])],'--', color = 'Grey')
    ax.plot([np.real(V2.pastpositions[0]), np.real(V3.pastpositions[0])], [np.imag(V2.pastpositions[0]), np.imag(V3.pastpositions[0])],'--', color = 'Grey')
    ax.plot([np.real(V1.pastpositions[0]), np.real(V3.pastpositions[0])], [np.imag(V1.pastpositions[0]), np.imag(V3.pastpositions[0])],'--', color = 'Grey')

    ax.plot([np.real(V1.pastpositions[-1]), np.real(V2.pastpositions[-1])], [np.imag(V1.pastpositions[-1]), np.imag(V2.pastpositions[-1])],'--', color = 'BLACK')
    ax.plot([np.real(V2.pastpositions[-1]), np.real(V3.pastpositions[-1])], [np.imag(V2.pastpositions[-1]), np.imag(V3.pastpositions[-1])],'--', color = 'BLACK')
    ax.plot([np.real(V1.pastpositions[-1]), np.real(V3.pastpositions[-1])], [np.imag(V1.pastpositions[-1]), np.imag(V3.pastpositions[-1])],'--', color = 'BLACK')

    ax.legend()
    fig.savefig('fig.png')

if Sim_type == 'Core Growth':
    for SaveD, fieldloc, cps in zip(vortexs.SaveD, vortexs.fieldPos, vortexs.critical_points):
        fig, (ax1, ax2) = plt.subplots(2,1, sharex = True)
        vortexs.Vort_peak = []
        vortexs.Saddles = []
        #vortexs.crit_points(SaveD[2], fieldloc[2], fieldloc[3])
        for peak in (vortexs.Vort_peak):
            ax2.plot(peak[0], peak[1], '.')
        #ax2.plot(vortexs.Saddles, 'o')
        ax2.plot([coord[0] for coord in cps], [coord[1]for coord in cps], 'o', color = 'red')
        #ax1.quiver(fieldloc[0], fieldloc[1], np.real(SaveD[0]), np.imag(SaveD[0]))
        ax1.streamplot(fieldloc[0], fieldloc[1], np.real(SaveD[0]), np.imag(SaveD[0]), density = 1.75)
        ax2.quiver(fieldloc[0], fieldloc[1], np.real(SaveD[2]), np.imag(SaveD[2]), scale = 0.5)
        ax2.contour(fieldloc[2], fieldloc[3], SaveD[2], 20, cmap = 'turbo')
        #ax2.streamplot(fieldloc[0], fieldloc[1], np.real(SaveD[2]), np.imag(SaveD[2]), density = 2)
        fig.savefig(f'Figures/StreamlinesatTime_{SaveD[1]}.png')
