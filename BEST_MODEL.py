import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize as opt

x, y1, zm = np.loadtxt('done1.txt', skiprows=1, unpack=True)
dim_distr=60

def model(z1, weight):
    y= [0 for i in range(len(z1))]
    zero = [0 for i in range(dim_distr)]
    z1_new= [z1[i] for i in range(len(z1))]
    for k in range(len(z1)):
        z_new= zero + z1_new[0:k]
        z_new.reverse()
        y[k]= np.dot(weight,z_new[0:dim_distr])
    return y
 
def DISPERSION(x):
    dist= [x[0]/((1+i)**x[1]) for i in range(dim_distr)]    
    real_data =92
    D= abs( model( zm , dist) - y1)
    DISP=[  (abs( D[i])**2.) for i in range(real_data)]
    return (sum(DISP)/len(DISP))**0.5

x0 = np.array([0.065, 0.41])
res = opt.minimize(DISPERSION, x0, method='Nelder-Mead')
print ('DISPERSION =',res['fun'])
print (res['x'])

kk= res['x'][0]; ll= res['x'][1]

print ('Dispersion =',DISPERSION([kk,ll])) 
dist= [kk/((1+i)**ll) for i in range(dim_distr)]
n=[i for i in range(dim_distr)]
print ('Sum distr =',sum(dist))
  
plt.plot (x, zm, label= 'input')
plt.plot (x, model( zm , dist), label='model')
plt.plot(x,y1, label ='outcome')
plt.title ('Model and actual data', fontsize = 22)
plt.xlabel ('Time, days', fontsize = 20)
plt.ylabel ('Number emails', fontsize =20)
plt.grid()
plt.legend()
plt.show()

labels = n
width = 0.50       # the width of the bars: can also be len(x) sequence
fig, ax = plt.subplots()
ax.bar(labels, dist, width,  label='liklyhood')
ax.set_ylabel('Visiting the website', fontsize = 15)
ax.set_xlabel('Number of emails', fontsize = 15)
ax.set_title('Distribution function',fontsize = 22)
ax.legend()
plt.show()

plt.plot(n, dist, 'ro')
plt.title ('Distribution function', fontsize = 22)
plt.xlabel ('Number of welcome emails', fontsize = 15)
plt.ylabel ('Visiting the website', fontsize =15)
plt.grid()
plt.show()