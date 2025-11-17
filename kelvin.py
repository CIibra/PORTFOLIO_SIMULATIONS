import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

nx, ny = 128, 128
Lx, Ly = 2.0, 2.0
dx, dy = Lx/nx, Ly/ny
dt = 0.005
nt = 200

x = np.linspace(0, Lx, nx, endpoint=False)
y = np.linspace(0, Ly, ny, endpoint=False)
X, Y = np.meshgrid(x, y)

u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
u[Y < Ly/2] = 1.0
u[Y >= Ly/2] = -1.0
v += 0.05*np.sin(2*np.pi*X/Lx)*np.exp(-((Y-Ly/2)/0.1)**2)

def compute_vorticity(u,v,dx,dy):
    dvdx = (np.roll(v,-1,axis=1)-np.roll(v,1,axis=1))/(2*dx)
    dudy = (np.roll(u,-1,axis=0)-np.roll(u,1,axis=0))/(2*dy)
    return dvdx-dudy

def compute_pressure(u,v):
    dudx = (np.roll(u,-1,axis=1)-np.roll(u,1,axis=1))/2
    dvdy = (np.roll(v,-1,axis=0)-np.roll(v,1,axis=0))/2
    return -(dudx+dvdy)

def advect(field,u,v,dx,dy,dt):
    f = np.copy(field)
    i_plus = np.roll(field,-1,axis=1); i_minus = np.roll(field,1,axis=1)
    j_plus = np.roll(field,-1,axis=0); j_minus = np.roll(field,1,axis=0)
    f -= dt*(
        np.where(u>0,u*(field-i_minus)/dx,u*(i_plus-field)/dx)+
        np.where(v>0,v*(field-j_minus)/dy,v*(j_plus-field)/dy)
    )
    return f

omega = compute_vorticity(u,v,dx,dy)
frames_vort, frames_press, frames_speed = [],[],[]
for t in range(nt):
    omega = advect(omega,u,v,dx,dy,dt)
    press = compute_pressure(u,v)
    speed = np.sqrt(u**2+v**2)
    if t%5==0:
        frames_vort.append(np.copy(omega))
        frames_press.append(np.copy(press))
        frames_speed.append(np.copy(speed))

fig,axes = plt.subplots(1,3,figsize=(15,5))
im_vort = axes[0].imshow(frames_vort[0],cmap='RdBu',origin='lower',extent=[0,Lx,0,Ly],vmin=-5,vmax=5)
axes[0].set_title("Vorticité")
im_press = axes[1].imshow(frames_press[0],cmap='viridis',origin='lower',extent=[0,Lx,0,Ly])
axes[1].set_title("Pression")
im_speed = axes[2].imshow(frames_speed[0],cmap='plasma',origin='lower',extent=[0,Lx,0,Ly])
axes[2].set_title("Vitesse")

def update(i):
    im_vort.set_array(frames_vort[i])
    im_press.set_array(frames_press[i])
    im_speed.set_array(frames_speed[i])
    return [im_vort,im_press,im_speed]

ani = animation.FuncAnimation(fig,update,frames=len(frames_vort),interval=100,blit=True)

output_path = "C:/Users/ADMIN/Desktop/kelvin_fields.gif"
folder = os.path.dirname(output_path)
if folder and not os.path.exists(folder): os.makedirs(folder)
ani.save(output_path,writer='pillow',fps=20)
print(" Animation enregistrée :",output_path)
