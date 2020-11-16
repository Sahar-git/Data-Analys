import matplotlib as mpl   
mpl.use('TkAgg')  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt                                        
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D

# Measured magnetic field for ES1 in the middle point of pinhole (H1)  
b11 = np.array([2, 2, 3, 4, 4, 5, 5, 6, 7, 6, 6, 5, 5, 4,
             21, 21, 24, 32, 39, 43, 46, 49, 50, 50, 51, 49, 45, 39,
             108, 108, 137, 170, 189, 201, 208, 212, 213, 213, 208, 201, 190, 171,
             124, 124, 173, 203, 221, 230, 237, 240, 240, 238, 231, 221, 212, 196,
             125, 125, 167, 201, 221, 233, 240, 244, 244, 241, 236, 226, 210, 182,
             126, 126, 158, 196, 219, 232, 240, 244, 245, 244, 239, 231, 217, 193,
             129, 129, 165, 202, 224, 236, 243, 246, 247, 245, 240, 231, 216, 192,
             129, 129, 180, 211, 230, 240, 246, 248, 249, 247, 243, 236, 221, 199,
             141, 141, 138, 186, 215, 232, 242, 248, 250, 250, 247, 241, 229, 209,
             132, 132, 182, 214, 232, 243, 249, 252, 252, 249, 244, 235, 220, 193,
             131, 131, 178, 191, 219, 234, 244, 249, 251, 250, 247, 241, 229, 211,
             112, 112, 150, 176, 190, 198, 201, 202, 202, 200, 197, 190, 179, 161]) 

# Measured magnetic field for ES1 at the higher step (H2)  
b12 = np.array([5, 6, 6, 7, 8, 8, 8, 9, 8, 8, 7, 6, 6, 
             37, 43, 54, 62, 68, 72, 74, 75, 78, 76, 73, 69, 61, 
             127, 147, 179, 198, 210, 218, 221, 222, 221, 215, 206, 190, 165, 
             140, 165, 197, 218, 231, 238, 241, 242, 239, 232, 222, 206, 178, 
             151, 154, 192, 216, 231, 239, 243, 245, 244, 239, 230, 217, 195, 
             171, 164, 201, 222, 235, 242, 246, 247, 245, 239, 230, 215, 189, 
             178, 158, 197, 221, 235, 240, 245, 248, 247, 244, 238, 227, 209, 
             181, 167, 203, 225, 238, 246, 249, 250, 248, 244, 237, 225, 203, 
             203, 180, 212, 231, 236, 245, 250, 252, 252, 248, 241, 229, 209, 
             218, 176, 207, 228, 241, 249, 252, 253, 252, 247, 239, 225, 201, 
             228, 184, 206, 227, 239, 247, 246, 249, 250, 249, 245, 238, 204, 
             148, 119, 134, 144, 149, 154, 164, 167, 153, 135, 130, 142, 131]) 

# Measured magnetic field for ES2 in the middle point of pinhole (H1)   
b21 = np.array([2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 3,  
            39, 55, 63, 69, 73, 75, 77, 77, 76, 72, 64, 56, 
            125, 157, 194, 207, 215, 220, 221, 221, 218, 213, 204, 190, 
            141, 180, 206, 221, 230, 235, 237, 236, 232, 226, 215, 199, 
            149, 188, 212, 226, 233, 237, 236, 234, 230, 222, 207, 184, 
            158, 190, 211, 225, 232, 235, 236, 236, 232, 225, 213, 193, 
            160, 196, 217, 230, 238, 242, 243, 241, 236, 227, 202, 171, 
            166, 198, 218, 231, 238, 242, 242, 241, 236, 226, 208, 181, 
            147, 182, 206, 221, 230, 235, 237, 237, 233, 226, 213, 192, 
            148, 181, 200, 210, 215, 218, 219, 218, 215, 208, 196, 174, 
            138, 161, 176, 185, 188, 189, 189, 188, 187, 183, 176, 161, 
            103, 110, 119, 123, 125, 125, 124, 123, 121, 118, 111, 99])  

# Measured magnetic field for ES2 at the higher step (H2)
b22 = np.array([1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 
             10, 13, 14, 16, 17, 17, 16, 16, 15, 13, 12, 10, 
             147, 154, 162, 166, 167, 167, 164, 160, 155, 146, 133, 111, 
             188, 205, 218, 225, 230, 231, 229, 225, 218, 206, 186, 154, 
             196, 214, 226, 234, 237, 238, 236, 231, 223, 208, 185, 150, 
             203, 217, 226, 231, 233, 234, 232, 227, 217, 201, 176, 154, 
             164, 196, 215, 227, 233, 236, 237, 235, 231, 228, 219, 202, 
             161, 178, 205, 222, 232, 238, 240, 241, 238, 232, 221, 204, 
             166, 167, 199, 218, 229, 235, 239, 240, 238, 232, 223, 208, 
             160, 162, 195, 214, 214, 224, 230, 233, 234, 232, 228, 220, 
             144, 170, 180, 180, 177, 175, 174, 173, 173, 173, 171, 167, 
             112, 137, 151, 153, 149, 146, 144, 143, 143, 144, 143, 139])

# Generation of a matrix, with entries from the magnetic field 
# measurement in electron spectrometer 1
def magneticField(x, y):
    return b11[x + 14*y//5]

x = np.arange(0, 14, 1)
y = np.arange(0, 56, 5)   

# Generation a mesh for later interpolation                                                     
xx, yy = np.meshgrid(x, y)
fval = magneticField(xx, yy)

# Plot of the measured magnetic fields
fig = plt.figure()
axis = Axes3D(fig)
axis.scatter(xx, yy, fval)

# Interpolation of the magnetic field measuring points     
f = interpolate.interp2d(x, y, fval, kind='cubic')
z = f(x, y)
surf = axis.plot_surface(xx, yy, z)

#Interpolation steps
a = np.linspace(0, 13, 14000)
b = np.linspace(0, 56, 12000)

# Interpolation of the magnetic field measuring points with a and b
g = f(a, b)

# defenition of another mesh
aa, bb = np.meshgrid(a, b)
fig = plt.figure()
axis = Axes3D(fig)
axis.scatter(aa, bb, g)

# CONSTANTS======================================================
E = 1000000 # Initial Energy of electron in ev                                         
dt = (0.2)*(10**(-14)) # Discrete time steps dt in seconds                 
MElectron = 511000 # mass of electron in (eV/c^2)
mElectron = (9.11)*(10**(-31)) # mass of electron in (kg)  
c0 = (3.)*(10**8) # Speed of light in vacuum in (m/s)       
e = (1.6)*(10**(-19)) # Eletrons charge in (C)
B1 = 0.25 # Nominal static magnetic field in T                                                                                
d = 0.00767 # Distance between pinhole and side IP in (m)                                                                
specLen = 0.05 # Length of the electron spectrometer1 in m (ES1)
gamma = 1 + (E/MElectron) # relativistic constant

# Electron velocity in (m/s)
vel = np.sqrt((np.power(c0, 2)) - (np.power(c0/gamma, 2))) 
# initial position
r0 = np.array([0, 0])         
# initial velocity                                        
v0 = np.array([0, vel])                     
# Initialization of the while loop
r = r0
v = v0
counter = 0
while((counter < 1000000) and (r[1] < specLen) and (abs(r[0]) < d)): 
    
    r = (r + (v*dt))  
    n = -int(((r[0] - 0.00017)/0.00001)//1)
    m = int((r[1]/0.00005)//1)
    B = g[11 - m][7 - n]/1000
    vsen=np.array([-v[1], v[0]]) # Vertical vector of initial velocity
    v = (v + dt*vsen*B*e/(gamma*mElectron)) # Actual electron velocity    
    w = abs(np.arctan(v[1]/v[0])) # Electron incidence angle
    counter = counter + 1
    
print(r)
print(v) 
print("Anzahl Schleifendurchläufe: ", counter)
if r[1] >= specLen:
    print("Elektron stoppt auf backIP bei ", (r[0] + d)*(10**3))
    print("Winkel auf die back IP in °", w)
    
if abs(r[0]) >= d:
    print("Elektron stoppt auf sideIP bei ", r[1]*(10**3))
    print("Winkel auf die back IP in °", w)
