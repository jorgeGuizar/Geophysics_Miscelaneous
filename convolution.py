import numpy as np

x=(2,2,1)
h=(2,-1)
nx=len(x)
nh=len(h)
yp=np.zeros(nx+nh-1)
for k in range(nx):
    for i in range(nh):
        yp[k+i]=yp[k+i]+x[k]*h[i]

