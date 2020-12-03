import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
import time
from numpy import matlib

dt = 10**-22
stop = 10**-16

Total_Length = 4.7*10**-6
Anode_Region = 1*10**-6
Cathode_Region = 1*10**-6
grid_shape = (Total_Length,10**-8,10**-8) # meters

s_y = grid_shape[1]
s_z = grid_shape[2]

t = np.arange(0,stop,dt)

virtual_chunks = 9


p_x = np.ones((1,1)) * (grid_shape[0] - np.random.rand() * Anode_Region)
p_y = np.ones((1,1)) * (np.random.rand() * grid_shape[1])
p_z = np.ones((1,1)) * (np.random.rand() * grid_shape[2])

a_x = np.ones((1,1)) * 0
a_y = np.ones((1,1)) * 0
a_z = np.ones((1,1)) * 0

v_x = np.ones((1,1)) * 0
v_y = np.ones((1,1)) * 0
v_z = np.ones((1,1)) * 0

Total_Particles = 1260

inactive_px = []
inactive_py = []
inactive_pz = []

for i in range(Total_Particles):
    inactive_px.append(grid_shape[0] - np.random.rand() * Anode_Region)
    inactive_py.append(np.random.rand() * grid_shape[1])
    inactive_pz.append(np.random.rand() * grid_shape[2])


m = 1.1525801 * 10**-28 # kg
Qi = 1.60217662 * 10**-19 # Coulombs
Qe = 1.60217662 * 10#**-19 # Coulombs
k = 8.99 * 10**9 # N m^2 / C^2
b = 10**-17
num_elec = 1

save_percentage = 0.01
save_rate = int(save_percentage*len(t))

pos = []
vel = []
tot = 0

p = 1

Reaction_Completion = len(p_x) / (Total_Particles + 1)

pot_diff = []

Anode_Potential = 3.01 # Volts
Cathode_Potential = 1 # Volts
alpha = 1
beta = 1
Cell_Potential = Anode_Potential * alpha + Cathode_Potential * beta # Volts

Recombined_Particles = 0

aap = []
bee = []

cathode_reaction_rate = 10000
anode_reaction_rate = 10000

freed_particles = len(p_x)
curr = []

for i in range(len(t)-1):

    if i % 1000 == 0:
        print(i/len(t))

        print('-------------')

    #current = (np.sum(-Qi*v_x/96485.3)/p)
    current = Qi * np.sum(v_x)
    zero_index = np.where(p_x <= Cathode_Region)[0]
    if (i % cathode_reaction_rate == 0 and len(zero_index) > 0 and len(p_x)>0): # removes 1 particle at a time

        #recombined_particle = p_x[np.argmin(p_x)]

        if (len(p_x)==0):
            break

        p_x = np.delete(p_x, np.argmin(p_x))
        p_y = np.delete(p_y, np.argmin(p_y))
        p_z = np.delete(p_z, np.argmin(p_z))

        v_x = np.delete(v_x, np.argmin(p_x))
        v_y = np.delete(v_y, np.argmin(p_y))
        v_z = np.delete(v_z, np.argmin(p_z))

        a_x = np.delete(a_x, np.argmin(p_x))
        a_y = np.delete(a_y, np.argmin(p_y))
        a_z = np.delete(a_z, np.argmin(p_z))

        p = len(p_x)

        p_x = p_x.reshape((p, 1))
        p_y = p_y.reshape((p, 1))
        p_z = p_z.reshape((p, 1))

        v_x = v_x.reshape((p, 1))
        v_y = v_y.reshape((p, 1))
        v_z = v_z.reshape((p, 1))

        a_x = a_x.reshape((p, 1))
        a_y = a_y.reshape((p, 1))
        a_z = a_z.reshape((p, 1))

        Recombined_Particles += 1

        beta = 1-(Recombined_Particles/Total_Particles)

        if p==1:
            break

    if i % anode_reaction_rate == 0 and i != 0 and len(inactive_px) > 0: # adds 1 particle at a time

        max_index = inactive_px.index(max(inactive_px)) # gets the particle nearest the electrolyte

        p_x = np.append(p_x, [inactive_px[max_index]])
        p_y = np.append(p_y, [inactive_py[max_index]])
        p_z = np.append(p_z, [inactive_pz[max_index]])

        v_x = np.append(v_x, [0])
        v_y = np.append(v_y, [0])
        v_z = np.append(v_z, [0])

        a_x = np.append(a_x, [0])
        a_y = np.append(a_y, [0])
        a_z = np.append(a_z, [0])

        p = len(p_x)
        print('ADDING A NEW PARTICLE', 1 - len(p_x) / (Total_Particles + 1))

        p_x = p_x.reshape((p, 1))
        p_y = p_y.reshape((p, 1))
        p_z = p_z.reshape((p, 1))

        v_x = v_x.reshape((p, 1))
        v_y = v_y.reshape((p, 1))
        v_z = v_z.reshape((p, 1))

        a_x = a_x.reshape((p, 1))
        a_y = a_y.reshape((p, 1))
        a_z = a_z.reshape((p, 1))

        del inactive_px[max_index]
        del inactive_py[max_index]
        del inactive_pz[max_index]

        freed_particles += 1

        alpha = 1 - (freed_particles / (Total_Particles + 1))

    Cell_Potential = alpha * Anode_Potential + beta * Cathode_Potential

    pot_diff.append(Cell_Potential)
    curr.append(current)

    aap.append(alpha)
    bee.append(beta)

    dist_x = np.matlib.repmat(p_x,1,p) - np.matlib.repmat(p_x,1,p).T
    dist_y = np.matlib.repmat(p_y,1,p) - np.matlib.repmat(p_y,1,p).T
    dist_z = np.matlib.repmat(p_z,1,p) - np.matlib.repmat(p_z,1,p).T

    y_shift_seed = np.array([ [0, 1,1,-1,-1,0,0,1,-1] ])
    y_bsize = np.ones((p,p)) * s_y
    y_shift = np.kron(y_shift_seed,y_bsize)

    z_shift_seed = np.array([ [0, 1,-1,-1,1,1,-1,0,0] ])
    z_bsize = np.ones((p,p)) * s_z
    z_shift = np.kron(z_shift_seed,z_bsize)

    #if ( p_x.all() == 0 ):

    #    print(p_x)

        # DO FINAL SAVE
    #    pos.append(p_x[0])

    #    print('EXITING ON LOOP',i)
    #    break  # if didn't break, would infect all data with NaN

    dist_x = np.matlib.repmat(dist_x,1,virtual_chunks)
    dist_y = np.matlib.repmat(dist_y,1,virtual_chunks) + y_shift
    dist_z = np.matlib.repmat(dist_z,1,virtual_chunks) + z_shift

    dist = np.sqrt( dist_x.copy()**2 + dist_y.copy()**2 + dist_z.copy()**2 )

    np.fill_diagonal(dist, 1.)
    np.fill_diagonal(dist_x, 1.)
    np.fill_diagonal(dist_y, 1.)
    np.fill_diagonal(dist_z, 1.)
    # necessary for calculation, see proof

    force = 1 / (dist ** 2)

    Fx = dist_x * force / dist
    Fy = dist_y * force / dist
    Fz = dist_z * force / dist

    np.fill_diagonal(Fx, 0)
    np.fill_diagonal(Fy, 0)
    np.fill_diagonal(Fz, 0)
    # necessary for calculation, see proof, undoes what is done three blocks above

    Fx = np.sum(Fx,1)
    Fx = Fx.reshape((p, 1))

    Fy = np.sum(Fy, 1)
    Fy = Fy.reshape((p, 1))

    Fz = np.sum(Fz, 1)
    Fz = Fz.reshape((p, 1))

    a_x = (-b * v_x + p*k*Qi*Qi* Fx - k*Qi*Qe*p/(p_x+0.001)**2
           - Qi * Cell_Potential / (2*Total_Length)) / m
    v_x = v_x + dt * a_x
    next_x = p_x + dt * v_x

    zero_index = np.where(next_x <= 0)[0]
    if len(zero_index) > 0:
        next_x[zero_index] = 0
        a_x[zero_index] = 0
        v_x[zero_index] = 0
    end_index = np.where(next_x >= Total_Length)[0]
    if len(end_index) > 0:
        next_x[end_index] = Total_Length
        a_x[zero_index] = a_x[zero_index]
        v_x[zero_index] = v_x[zero_index]
    p_x = next_x

    a_y = (-b * v_y + k*Qi*Qi* Fy) / m
    v_y = v_y + dt * a_y
    p_y = (p_y + dt * v_y) % grid_shape[1]

    a_z = (-b * v_z + k*Qi*Qi* Fz) / m
    v_z = v_z + dt * a_z
    p_z = (p_z + dt * v_z) % grid_shape[2]

    #if i%save_rate == 0:

    # SAVE THE DATA EACH SAVE PERCENTAGE OF THE TIME

    #print(i/len(t))

plt.plot(curr,pot_diff)
plt.ylabel('Voltage (V)')
plt.xlabel('Capacity (A)')
plt.title('LiPON Discharge Curves')
plt.show()