
import numpy as np
import matplotlib as mpl   
mpl.use('TkAgg') 
import matplotlib.pyplot as plt                                        
#From mpl_toolkits.mplot3d import Axes3D

# Background data for back-on IP and also side-on IP
background_data = ['ES1_back_background_44.txt', 'ES1_side_background_44.txt']

# Signal data for back-on IP and also side-on IP
signal_data = ['ES1_back_signal_44.txt', 'ES1_side_signal_44.txt']

# Number of pixels of background which were cut off
pixel_backip = np.linspace(1, 311, 311)
# Number of pixels of signal which were cut off
pixel_sideip = np.linspace(1, 1119, 1119)

# Data of the electron energy dispersion fit, first entry correspond to the 
# back-on IP and the second entry to the side-on IP values
a = [-8.67051*(10**(-4)), -5.4232*(10**(-4))]
b = [0.92522, -1.22626]
c = [1.56064*(10**(-4)), 0.00502]
y0 = 0.0267
A1 = 0.14642
t1 = 0.44388

# Steps of creation an integrated electron signal
for k in range(0, 2):
    background = np.genfromtxt(background_data[k])
    s = background.shape
    column = np.zeros(s[1])
    for i in range(s[1]):
        column[i] = np.mean(background[:, i])
    site = np.genfromtxt(signal_data[k])
    s2 = site.shape
    column2 = np.zeros(s2)
    for i in range(s2[1]):
        column2[:, i] = site[:, i] - column[i]
    signal = column2.sum(axis = 0)
    print(k)
    print(signal)

    # Steps of obtining dN/dE for back-on IP 
    if k == 0:
        # Equation of the electron energy energy dispersion obtained by fitting in MeV
        energy_backip = (b[k]+(c[k]*(pixel_backip+1000)))/(1 + a[k]*(pixel_backip + 1000)) 
        
        # 1/(dE/dx)       
        dedx_backip = ((1 + ((a[k]*(pixel_backip + 1000))))**2)/(c[k] - (a[k]*b[k])) 

        # Equation of stopping power obtaibed by fitting
        stopingpower_backip = (1.67914 + (0.02316*(energy_backip)))/(1.67914 + (0.02316*10))

        # applying the stopping power for electrons with energies E> 10 MeV
        idx_stop = [idx for idx in range(0, len(energy_backip)) if energy_backip[idx] <= 10]
        stopingpower_backip[idx_stop] = 1

        #Equation of relative sensitivity of the IPs obtained by fitting
        sensitivity_backip = A1*np.exp(-energy_backip/t1) + y0

        # Steps of dN/dE
        electron_backip = signal/sensitivity_backip
        angle_back = (-141.91+(0.12304*(pixel_backip + 1000)))
        relativ_sensit_back= 1.15/np.cos(angle_back*0.0164) - 0.15
        dnde_sensitive_backip = dedx_backip*electron_backip
        dnde_stoping_backip= dnde_sensitive_backip/stopingpower_backip
        dnde_relativ_sensit_back = dnde_stoping_backip/relativ_sensit_back
        print(dedx_backip)
        print(energy_backip)
        
    # Steps of obtaining dN/dE for back-on IP 
    else:
        energy_sideip = (b[k] + c[k]*pixel_sideip)/(1 + a[k]*pixel_sideip)
        dedx_sideip = (a[k] - b[k]*c[k])/(a[k] + c[k]*pixel_sideip)**2
        dedx_sideip = ((1+(a[k]*pixel_sideip))**2)/(c[k] - (a[k]*b[k])) 
        stopingpower_sideip = 1.67914 + ((0.02316*(energy_sideip)))//(1.67914 + (0.02316*10))
        idx_stop = [idx for idx in range(0, len(energy_sideip)) if energy_sideip[idx] <= 10]
        stopingpower_sideip[idx_stop] = 1
        sensitivty_sideip = A1*np.exp(-energy_sideip/t1) + y0
        electron_sideip = signal/sensitivty_sideip
        angle_side = ((-117.89292*np.exp(-pixel_sideip/134.38423)) + 
                      ((-61.9476)*np.exp(-pixel_sideip/531.56658)) + 80.65)
        relativ_sensit_side = 1.15/np.cos(angle_side*0.0164) - 0.15
        dnde_sensitive_sideip = dedx_sideip*electron_sideip
        dnde_stoping_sideip = dnde_sensitive_sideip/stopingpower_sideip
        dnde_relativ_sensit_side = dnde_stoping_sideip/relativ_sensit_side
        print(dedx_sideip)
        print(electron_sideip)

# Concatenation of both IPs
energy = np.concatenate([energy_sideip, energy_backip])
dnde_sensitive = np.concatenate([dnde_sensitive_sideip, dnde_sensitive_backip])
dnde_stoping = np.concatenate([dnde_stoping_sideip, dnde_stoping_backip])
dnde_relativ_sensit = np.concatenate([dnde_relativ_sensit_side, dnde_relativ_sensit_back])
pixeltotal = np.concatenate([pixel_sideip, pixel_backip + 1000])
plt.figure()
axis = plt.gca()
axis.plot(energy, dnde_stoping)
axis.plot(energy, dnde_relativ_sensit, 'o')
plt.figure()
plt.plot(energy_sideip, relativ_sensit_side)
np.savetxt("ES1, d = 0.00767", dnde_relativ_sensit)
data = np.array([energy, dnde_relativ_sensit])
data = data.T
np.savetxt("dnde_relativ_sensit, d = 0.00767", data)

# Integrated signal graphic of side
plt.figure()
plt.plot(pixel_sideip ,signal)  
 
                
