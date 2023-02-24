import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define constants
G = 6.67430e-11  # gravitational constant
m_planet = 5.972e24  # mass of the planet (kg)
r_planet = 6.371e6  # radius of the planet (m)
altitude = 408e3  # altitude of the satellite above the planet's surface (m)
m_satellite = 1000  # mass of the satellite (m)
v_init = np.array([0, 7660, 0000])  # initial velocity of the satellite (m/s)

# Define initial position of the satellite (m)
r_init = np.array([r_planet + altitude, 0, 0])

# Define time step (s)
dt = 1

# Define simulation duration (s)
duration = 1 * 2 * np.pi * np.sqrt(((r_planet + altitude) ** 3) / (G * m_planet))

# Initialize arrays to store position and velocity
r = np.zeros((int(duration / dt), 3))
v = np.zeros((int(duration / dt), 3))
t = np.zeros(int(duration / dt))
a = np.zeros(int(duration / dt))

# Set initial position and velocity
r[0] = r_init
v[0] = v_init
t[0] = 0
a[0] = abs( ( (a[0]-r[0][0])**2 + (a[0]-r[0][1])**2 + (a[0]-r[0][2])**2)**(1/2) - r_planet)
# a[0] = altitude

# Calculate position and velocity at each time step using Euler's method
for i in range(1, len(t)):
    # Calculate acceleration due to gravity
    r_mag = np.linalg.norm(r[i-1])
    a_gravity = -G * m_planet / r_mag ** 3 * r[i-1]
    
    # Update velocity
    v[i] = v[i-1] + a_gravity * dt
    
    # Update position
    r[i] = r[i-1] + v[i] * dt
    
    # Update time
    t[i] = t[i-1] + dt
    
    # Update altitude
    a[i] = abs( ( (a[0]-r[i][0])**2 + (a[0]-r[i][1])**2 + (a[0]-r[i][2])**2)**(1/2) - r_planet)

    # break loop if crashes into surface
    if a[i] < 0:
        print('Catastrophic failure due to high velocity physics interaction')
        break

# Create 3D plot of the satellite's orbit
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  

# Create corner points to ensure the axis are equal (matplotlib doesn't have this capability in 3D!)
x = [i[0] for i in r]
y = [i[1] for i in r]
z = [i[2] for i in r]

max_value = max(max(x),max(y),max(z))
min_value = min(min(x),min(y),min(z))

ax.plot(max_value, max_value, max_value)
ax.plot(min_value, min_value, min_value)

# Create a sphere to represent the planet
u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
x = r_planet * np.cos(u) * np.sin(v)
y = r_planet * np.sin(u) * np.sin(v)
z = r_planet * np.cos(v)
ax.plot_surface(x, y, z, cmap = 'gray', alpha=0.5)

# Plot Orbit
ax.plot(r[:, 0], r[:, 1], r[:, 2])

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Satellite orbiting planet')
plt.show()