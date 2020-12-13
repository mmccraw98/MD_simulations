import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq


print(np.where(np.diff(potentials[2000:]) > 0.1))
import pickle
with open(r'C:\Users\mmccraw98\Downloads\dz.pkl', 'rb') as f:
     d, f = pickle.load(f)
start = 350
d, f = d[start:np.argmax(f)], f[start:np.argmax(f)]
#f -= f[np.argmin(d)]
f -= f[22]
d -= d[0]
fig = plt.gcf()
plt.plot(f*10**9, d*10**9)
plt.xlabel('Distance Between Cantilever Base and Surface (nm)')
plt.ylabel('Cantilever Deflection (nm)')
plt.grid()
plt.show()

potentials = pd.read_csv(r'C:\Users\mmccraw98\AppData\Local\LAMMPS 64-bit 29Oct2020\Examples\TESTS\Marshall_McCraw_CN_Final_Project\potentials.csv').values[5700:]
index1 = 1000
index2 = 700
p3 = potentials[7000+np.argmax(potentials[7000:]):]
p1 = potentials[np.argmax(potentials[:2000]):p3.size]
p2 = potentials[3000+np.argmax(potentials[3000:6000]):3000+p3.size]

p1 = p1[index1:]
p2 = p2[index1:]
p3 = p3[index1:-index2]

print(np.argmin(p1))
print(np.argmin(p2))
print(np.argmin(p3))

fig = plt.gcf()
plt.plot(p1, label='Pt-Pt')
plt.plot(p2, label='Pt-Au')
plt.plot(p3, label='Au-Au')
plt.grid()
plt.xlabel('Interatomic Distance (A)')
plt.ylabel('Potential Energy (eV)')
plt.legend()
plt.show()
fig.savefig(r'C:\Users\mmccraw98\Downloads\ptau.png')



center = np.mean(points, axis=0)
dist = np.mean(np.abs(points - center), axis=0)
print(np.sqrt(dist[0]**2+dist[1]**2))

points = np.array([np.array([27.718599, 27.205700]),
                   np.array([24.946699, 27.205700]),
                   np.array([23.560801, 29.606199]),
                   np.array([22.174900, 32.006699]),
                   np.array([23.560801, 34.407200]),
                   np.array([24.946699, 36.807701]),
                   np.array([27.718599, 36.807701]),
                   np.array([29.104500, 34.407200]),
                   np.array([30.490400, 32.006699]),
                   np.array([29.104500, 29.606199])])

center = np.mean(points, axis=0)
dist = np.mean(np.abs(points - center), axis=0)
print(np.sqrt(dist[0]**2+dist[1]**2))

file = open(r'C:\Users\mmccraw98\AppData\Local\LAMMPS 64-bit 29Oct2020\Examples\TESTS\CN_FINAL_APPROACH\final_data.txt', 'r')
raw_data = file.read()
file.close()

approach, retract = raw_data.split(sep='RETRACT')
a_step, a_d, a_f = [], [], []
for line in approach.split(sep='\n')[1:]:
    print(line.split(sep='    '))
    print(line.split(sep='   '))
idx = np.argmax(f)
I1, I2 = 150, 378
df = f[:idx][I1:][np.argmin(d[:idx][I1:])]*10**9
fig = plt.gcf()
plt.plot(f[:idx][I1:]*1*10**9-df, d[:idx][I1:]*10**9, label='Approach')
plt.plot(f[idx:][:I2]*1*10**9-df, d[idx:][:I2]*10**9, color='r', label='Retract')
plt.grid()
plt.xlabel('Distance (nm)')
plt.ylabel('Force (nN)')
plt.legend()
plt.show()
fig.savefig(r'C:\Users\mmccraw98\Downloads\fd_2.png')
path = r'C:\Users\mmccraw98\AppData\Local\LAMMPS 64-bit 29Oct2020\Examples\TESTS\Marshall_McCraw_CN_Final_Project\data.csv'
data = pd.read_csv(path)
step, TotEng, dz, fz = data.Step.values, data.TotEng.values, data.v_tipxz.values, data.v_tipfz.values
fc = 0.01
b = 0.08
N = int(np.ceil(4/b))
N += N % 2
n = np.arange(N)

sinc = np.sinc(2*fc*(n-(N-1)/2))
window = 0.42 - 0.5 * np.cos(2*np.pi*n/(N-1)) + 0.08*np.cos(4*np.pi*n/(N-1))
sinc *= window
sinc /= np.sum(sinc)

final = np.convolve(fz, sinc)
idx3 = (int(abs(dz.size-final.size)/2))
final = final[idx3:-(idx3+1)]
print(min(final))
fig = plt.gcf()
idx = np.argmax(final[:int(fz.size/2)])
plt.plot(dz[idx:], final[idx:], color='r', label='Retract')
plt.plot(dz[:idx], final[:idx], label='Approach')
plt.plot(dz)
print(np.min(dz))
plt.grid()
plt.legend()
plt.xlabel('Distance (A)')
plt.ylabel('Force (nN)')
plt.show()
fig.savefig(r'C:\Users\mmccraw98\Downloads\fd_3.png')
plt.plot(step*0.001, fz, label='Raw')
plt.plot(step*0.001, final, color='r', label='Filtered')
plt.legend()
plt.grid()
plt.xlabel('Time (ps)')
plt.ylabel('Force (nN)')
plt.show()
fig.savefig(r'C:\Users\mmccraw98\Downloads\f4.png')