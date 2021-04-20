import textwrap
import numpy as np
from scipy.integrate import odeint
from bisect import bisect_left as bsearch

# Global constants

# Lorenz paramters and initial conditions
a, b, c = 10, 2.667, 28  # sigma = 10, ro = 28 and beta = 2.667 for Lorenz system
x0, y0, z0 = 0, 0, 0

# Maximum time point and total number of time points
tmax, N = 100, 10000

# the three formulas for lorrenz -> dx/dt, dy/dt, dz/dt
def lorenz(X, t, a, b, c):

    x, y, z = X

    # the three formulas for lorrenz 
    x_dot = -a*(x - y)   # dx/dt
    y_dot = c*x - y - x*z     # dy/dt
    z_dot = -b*z + x*y    # dz/dt
    return x_dot, y_dot, z_dot

# uses the key to generate random x0, y0, z0
def update_lorenz (key):

    # bin(val) method returns the binary string equivalent to the val(base10)
    # int(val, base) converts val to base 10
    # so, hex key -> integer -> binary
    key_bin = bin(int(key, 16))[2:].zfill(256)  #covert hex key digest to binary
    
    # key_bin has length 256 
    #// key_bin: 00110011....000100

    k={}                                        # key dictionary
    key_32_parts=textwrap.wrap(key_bin, 8)      # slicing key into 8 parts
    # Wraps the key_bin so every sub-key is at most 8 characters long. 
    #// key_32_parts: ['00110011', '00010011',..... '00110010']

    # giving numbers to the 32 parts as k1, k2 ...
    num=1
    for i in key_32_parts:
        k["k{0}".format(num)]=i
        num = num + 1
    #// k: {'k1': '00110011', 'k2': '00010011',..'k32': '01110000'}

    t1 = t2 = t3 = 0
    # from k1 to k12 convert them from binary to base 10
    # then t1 = (0 xor k1) = k1
    # then t2 = (t1 xor k2) = (k1 xor k2) .... and so on
    # basically chain XORing
    for i in range (1,12):
        t1=t1^int(k["k{0}".format(i)],2)

    for i in range (12,23):
        t2=t2^int(k["k{0}".format(i)],2)

    for i in range (23,33):
        t3=t3^int(k["k{0}".format(i)],2) 
    
    #// t1=2 t2=210 t3=203
    global x0 ,y0, z0   #// at this point, x0=y0=z0 is 0
    x0=x0 + t1/256           
    y0=y0 + t2/256         
    z0=z0 + t3/256
    #// x0=0.0078125, y0=0.8203125, z0=0.79296875
    # hence initial values of x,y,z are set randomly

# plots the Lorenz graph
def plot(x,y,z):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    s = 100
    c = np.linspace(0,1,N)
    for i in range(0,N-s,s):
        ax.plot(x[i:i+s+1], y[i:i+s+1], z[i:i+s+1], color=(1-c[i],c[i],1), alpha=0.4)
    plt.show()

# generates the lorenz chaotic sequence
def gen_chaos_seq(m,n):
    # the lorenz parameters
    global x0,y0,z0,a,b,c,N
    N=m*n*4

    # initialize empty x, y, z arrays
    x= np.array((m,n*4))
    y= np.array((m,n*4))
    z= np.array((m,n*4))
    #// x = [600 1560]

    # numpy.linspace(start, stop, divisions)
    t = np.linspace(0, tmax, N) 
    #// N = 936000 -> so 0 to 100 divided into 936000 parts
    #// t = [0.00000000e+00 1.0621e-04, 2.13442e-04 ... 9.997863e+01 , 9.99998932e+01 1.00000000e+02] len-936000
    
    # scipy.integrate.odeint -> Integrate a system of ordinary differential equations.
    # scipy.integrate.odeint(func, initial conditions, t, args=())
    # lorenz gets the initial values, time and parameters
    # lorenz gets called 27620 times. Each time derivative returned, integrated, and then sent again as x0,y0,z0
    f = odeint(lorenz, (x0, y0, z0), t, args=(a, b, c))
    #// f = [[7.81250000e-03 8.20312500e-01 7.92968750e-01]...[]]
    #// f.shape -> (936000, 3)

    # x,y,z are transpose of f, and each column distributed
    x, y, z = f.T
    #//x.shape = (936000,)

    x=x[:(N)]
    y=y[:(N)]
    z=z[:(N)]
    #//x.shape = (936000,)
    return x,y,z

# fx[i] holds the index of where x[i] belongs in the sorted order of x
def sequence_indexing(x,y,z):
    n=len(x)
    fx=np.zeros((n),dtype=np.uint32)
    fy=np.zeros((n),dtype=np.uint32)
    fz=np.zeros((n),dtype=np.uint32)
    
    #// n => 936000
    # bsearch(a,x) -> Locate the insertion point for x in a to maintain sorted order.
    # If x is already present in a, it returns index of the place it is found

    sorted_x=sorted(x)
    for k1 in range(0,n):
        t = x[k1]
        # t is extremely small eg. 3.357436706831494
        k2 = bsearch(sorted_x, t)
        # k2 is the index of place where we find it in sorted_x
        # eg. 204361
        fx[k1]=k2
        # fx[i] holds the index of where x[i] belongs in the sorted order of x
        # // x = [33, 47, 64, 79, 21, 170, 87, 63, 20, 21]
        # // sorted_x = [20, 21, 21, 33, 47, 63, 64, 79, 87, 170]
        #// x[2] = 64. where does 64 belong in sorted order? index 6. hence fx[2] is 6
        #// fx = [3 4 6 7 1 9 8 5 0 1]

    sorted_y=sorted(y)
    for k1 in range(0,n):
        t = y[k1]
        k2 = bsearch(sorted_y, t)
        fy[k1]=k2

    sorted_z=sorted(z)
    for k1 in range(0,n):
        t = z[k1]
        k2 = bsearch(sorted_z, t)
        fz[k1]=k2
    #// fx = [471655 471686 471717 ... 532089 532144 532200] 
    return fx,fy,fz
        