# https://github.com/tafseerahmed/image-Encryption-dna-encoding/blob/master/encr.py
# Aim: create a image encryption program using Dna encoding and chaos map

from PIL import Image
import tkinter as tk
from tkinter import filedialog
import hashlib 
import binascii
import textwrap
import cv2
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from importlib import reload  
from bisect import bisect_left as bsearch

# Global constants

# Lorenz paramters and initial conditions
a, b, c = 10, 2.667, 28  # sigma = 10, ro = 28 and beta = 2.667 for Lorenz system
x0, y0, z0 = 0, 0, 0

#DNA-Encoding RULE => A = 00, T=01, G=10, C=11
dna={}
dna["00"]="A"
dna["01"]="T"
dna["10"]="G"
dna["11"]="C"

# for DNA decoding
dna["A"]=[0,0]
dna["T"]=[0,1]
dna["G"]=[1,0]
dna["C"]=[1,1]

#DNA xor
dna["AA"]=dna["TT"]=dna["GG"]=dna["CC"]="A" #// example 00 xor 00 => 00
dna["AG"]=dna["GA"]=dna["TC"]=dna["CT"]="G" #// example 01 xor 11 => 10
dna["AC"]=dna["CA"]=dna["GT"]=dna["TG"]="C" #// example 00 xor 11 => 11
dna["AT"]=dna["TA"]=dna["CG"]=dna["GC"]="T" #// example 10 xor 11 => 01

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

# input image file name from user
def image_input_from_user():                           #returns path to selected image
    path = "NULL" 
    path = input("Please enter file name:  ")         # show an "Open" dialog box and return the path to the selected file
    if path!="NULL":
        print("Image loaded!") 
    else:
        print("Error Image not loaded!")
    return path

# input key from user
def key_input_from_user():
    key = input("Enter key: ")
    return key

# given an image, convert to R, G, B arrays
def split_into_rgb_channels(image):
    # BGR values
    #// image.shape (390, 600, 3) => width * height * channels
    red = image[:,:,2] # each pixel, index2
    green = image[:,:,1] # each pixel, index1
    blue = image[:,:,0] # each pixel, index0
    # red array takes out R values from each pixel (red = [[139 18 ... 37], [142 141 ..130]...[....]]) 
    # len(red) is width of image, len(red[0] is height of image)
    return red, green, blue

# returns key, width and height of image
def securekey (file_path, key):
    #// key: abc123
    img = Image.open(file_path)
    m, n = img.size

    # // pixels: 234000  m: 390 n: 600 
    # m is width, n is height
    print("pixels: {0}  width: {2} height: {1} ".format(m*n, m, n))
    pix = img.load()                # pix[0, 0] gives a tuple (B,R,G)
    plainimage = list()                  
    # n is height (no of rows) (goes from L->R)
    for y in range(n):
        for x in range(m):
            for k in range(0,3):
                plainimage.append(pix[x,y][k])   
    # plainimage contains all the rgb values continuously
    #// plainimage: [120, 208, 222, 119, ....205, 218, 123 ] length = n*m*3
    
    # encode() : Converts the string into bytes to be acceptable by hash function. (binary) 
    key_in_bytes = key.encode()
    #// key_in_bytes: binary form of abc123

    # key is made a hash.sha256 object -> length 32
    key = hashlib.sha256()                        

    # image data is fed to generate digest
    # key.update(bytearray(plainimage)) 

    # hexdigest() : Returns the encoded data in hexadecimal format -> length 64
    key_hex_digest = key.hexdigest()
    #// key_hex_digest: 331332a.....0b0670
    return  key_hex_digest, m, n

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

# image converted to three matrices of R, G, B colors
def decompose_matrix(file_path):
    image = cv2.imread(file_path)
    #// image.shape (390, 600, 3) => width * height * channels
    blue,green,red = split_into_rgb_channels(image)
    # now, blue, red, green are array of arrays. To access elements -> blue[0][0] => <class 'numpy.ndarray'>
    # blue, red, green are all 'm' length arrays each containing n elements

    # BGR is the order, hence 0->blue, 1->green, 2->red
    for values, channel in zip((red, green, blue), (2,1,0)):
        img = np.zeros((values.shape[0], values.shape[1]), dtype = np.uint8)
        img[:,:] = (values)
        if channel == 0:
            B = np.asmatrix(img)
        elif channel == 1:
            G = np.asmatrix(img)
        else:
            R = np.asmatrix(img)
    # now B, R, G are matrices -> to access elements, B[0,0]  => <class 'numpy.matrix'>
    return B,G,R

# convert red,blue,green matrices decimals -> bits -> DNA encoded letters
def dna_encode(b,g,r):
    
    #// b.shape => (390, 600)
    # each number converted to 8 bits format
    b = np.unpackbits(b,axis=1)
    g = np.unpackbits(g,axis=1)
    r = np.unpackbits(r,axis=1)
    #// b.shape => (390, 4800)

    m,n = b.shape   # m is width, n is height*8
    # numpy.chararray(shape)

    # initialise empty array for DNA encoded -> so that height 2 bits converted to 1 letter
    r_enc= np.chararray((m,int(n/2)))
    g_enc= np.chararray((m,int(n/2)))
    b_enc= np.chararray((m,int(n/2)))
    #// b_enc.shape => (390, 2400)
    
    # for each channel (matrix)
    for color,enc in zip((b,g,r),(b_enc,g_enc,r_enc)):
        idx=0
        # m-> entire width
        for j in range(0,m):
            # n -> height*8, i jumps alternately
            for i in range(0,n,2):
                # enc[] dna["11"] will be C
                enc[j,idx]=dna["{0}{1}".format(color[j,i],color[j,i+1])]
                idx+=1
                # final bit traverse, goto outer loop
                if (i==n-2):
                    idx=0
                    break
    
    # b.shape => (390, 4800)
    # b_enc.shape => (390, 2400) (2 bits converted to a letter)
    # b_enc = [[b'G' b'A' b'G' ... b'G' b'T' b'T']...[]] => hence the bits are encoded in the form of A,G,T,C
    b_enc=b_enc.astype(str)
    g_enc=g_enc.astype(str)
    r_enc=r_enc.astype(str)
    # [['G' 'A' 'G' ... 'G' 'T' 'T']..['A' 'A' 'A' ... 'A' 'A' 'A']]
    return b_enc,g_enc,r_enc

# encode key matrix using chaos of b matrix, DNA encoding
def key_matrix_encode(key,b):    
    
    # b decimal -> binary numbers
    #// b.shape => (390, 600)
    b = np.unpackbits(b,axis=1)
    #// b.shape => (390, 4800)
    m,n = b.shape
    #// m = 390, n = 4800

    #// key= 3313329136...5d0b0670 (length 64)
    # convert hexadecimal to decimal, remove 0b and fill the MSB with 0
    key_bin = bin(int(key, 16))[2:].zfill(256)
    #// key_bin => 0b11001100....00000 (length 256) -> 0011001100....00000

    Mk = np.zeros((m,n),dtype=np.uint8)
    # Mk matrix => width x height*8

    x=0
    # x takes value of 0,1,2,3 ....1872000
    # x%256 takes value of 0,1,2, ...255
    # key_bin[1 to 255] will give any of 1 or 0 (hence randomness generated in Mk)
    for j in range(0,m):
        for i in range(0,n):
            Mk[j,i]=key_bin[x%256]
            x+=1
    #// Mk  = [[0 0 1 ... 0 0 1]...[0 0 0 ... 0 1 0]] shape-(390, 4800)

    # encrypting the bits with DNA letters A, C, G and T
    Mk_enc=np.chararray((m,int(n/2)))
    idx=0
    for j in range(0,m):
        for i in range(0,n,2):
            if idx==(n/2):
                idx=0
            Mk_enc[j,idx]=dna["{0}{1}".format(Mk[j,i],Mk[j,i+1])]
            idx+=1
    #// Mk_inc = [[b'A' b'C' b'A' ... b'A' b'A' b'T']...[]] shape-(390,2400)
    Mk_enc=Mk_enc.astype(str)
    #// Mk_inc = [['A' 'C' 'A' ... 'A' 'A' 'T']...[]] 
    return Mk_enc

# xor the color matrices with key matrix
def xor_operation_encrypt(b,g,r,mk):
    m,n = b.shape
    #// m = 390, n = 2400

    # initialize bx, gx, rx
    bx=np.chararray((m,n))
    gx=np.chararray((m,n))
    rx=np.chararray((m,n))
    #// bx = [['', '', '', '', '', ...]...[]]

    b=b.astype(str)
    g=g.astype(str)
    r=r.astype(str)
    #// b = ['G' 'A' 'G' ... 'G' 'T' 'T']..[]]

    # final matrix = color matrix element(DNA) ^ Mkey element(DNA)
    for i in range(0,m):
        for j in range (0,n):
            bx[i,j] = dna["{0}{1}".format(b[i,j],mk[i,j])]
            gx[i,j] = dna["{0}{1}".format(g[i,j],mk[i,j])]
            rx[i,j] = dna["{0}{1}".format(r[i,j],mk[i,j])]
    #// bx = [[b'G' b'C' b'G' ... b'G' b'T' b'A']..[]]

    bx=bx.astype(str)
    gx=gx.astype(str)
    rx=rx.astype(str)
    #// [['G' 'C' 'G' ... 'G' 'T' 'A']...[]]
    return bx,gx,rx 

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

# plots the Lorenz graph
def plot(x,y,z):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    s = 100
    c = np.linspace(0,1,N)
    for i in range(0,N-s,s):
        ax.plot(x[i:i+s+1], y[i:i+s+1], z[i:i+s+1], color=(1-c[i],c[i],1), alpha=0.4)
    plt.show()

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
        
# scramble the blue, green, red arrays wrt fx,fy and fz
def scramble_encrypt(fx,fy,fz,b,r,g):
    
    p,q=b.shape
    #// p = 390, q= 2400, b = [['G' 'C' 'G' ... 'G' 'T' 'A']..[]]
    size = p*q

    # open up the 2d array into single one of p*q
    bx=b.reshape(size).astype(str)
    gx=g.reshape(size).astype(str)
    rx=r.reshape(size).astype(str)
    #// bx = ['G' 'C' 'G' ... 'C' 'A' 'G']

    # empty array of p*q
    bx_s=np.chararray((size))
    gx_s=np.chararray((size))
    rx_s=np.chararray((size))

    # Z -> Blue, Y -> Green, X -> Red
    # take each element from fz, we get index, hence, 
    # bx[that_index] is assigned to bx_s array
    for i in range(size):
        idx = fz[i]
        bx_s[i] = bx[idx]
    for i in range(size):
        idx = fy[i]
        gx_s[i] = gx[idx]
    for i in range(size):
        idx = fx[i]
        rx_s[i] = rx[idx]    

    #// bx_s = [b'C' b'A' b'T' ... b'C' b'A' b'G']
    bx_s=bx_s.astype(str)
    gx_s=gx_s.astype(str)
    rx_s=rx_s.astype(str)
    #// bx_s = ['C' 'A' 'T' ... 'C' 'A' 'G']
    
    # empty matrix of size p*q
    b_s=np.chararray((p,q))
    g_s=np.chararray((p,q))
    r_s=np.chararray((p,q))

    # fold bx_s to p*q
    b_s=bx_s.reshape(p,q)
    g_s=gx_s.reshape(p,q)
    r_s=rx_s.reshape(p,q)
    #// b_s = [['C' 'A' 'T' ... 'G' 'T' 'A']...[]]
    return b_s,g_s,r_s

def scramble_decrypt(fx,fy,fz,b,g,r):
    p,q=b.shape
    size = p*q
    bx=b.reshape(size)
    gx=g.reshape(size)
    rx=r.reshape(size)

    bx_s=b.reshape(size)
    gx_s=g.reshape(size)
    rx_s=r.reshape(size)
    
    bx=bx.astype(str)
    gx=gx.astype(str)
    rx=rx.astype(str)
    bx_s=bx_s.astype(str)
    gx_s=gx_s.astype(str)
    rx_s=rx_s.astype(str)
    
    for i in range(size):
            idx = fz[i]
            bx_s[idx] = bx[i]
    for i in range(size):
            idx = fy[i]
            gx_s[idx] = gx[i]
    for i in range(size):
            idx = fx[i]
            rx_s[idx] = rx[i]    

    b_s=np.chararray((p,q))
    g_s=np.chararray((p,q))
    r_s=np.chararray((p,q))

    b_s=bx_s.reshape(p,q)
    g_s=gx_s.reshape(p,q)
    r_s=rx_s.reshape(p,q)

    return b_s,g_s,r_s

# get DNA letters A,C,G,T back to bits (so image original size maintianed)
def dna_decode(b,g,r):
    m,n = b.shape
    # n is half of original image since we combined two bits to one letter
    r_dec= np.ndarray((m,int(n*2)),dtype=np.uint8)
    g_dec= np.ndarray((m,int(n*2)),dtype=np.uint8)
    b_dec= np.ndarray((m,int(n*2)),dtype=np.uint8)

    # for all three channels, replace the letter A,T,G,C with 00,01,10,11 respectively
    for color,dec in zip((b,g,r),(b_dec,g_dec,r_dec)):
        for j in range(0,m):
            for i in range(0,n):
                dec[j,2*i]=dna["{0}".format(color[j,i])][0]
                dec[j,2*i+1]=dna["{0}".format(color[j,i])][1]

    #// b_dec = [[1 1 0 ... 1 0 0] ...[]]
    b_dec=(np.packbits(b_dec,axis=-1))
    g_dec=(np.packbits(g_dec,axis=-1))
    r_dec=(np.packbits(r_dec,axis=-1))
    #// b_dec = [[198 110 254 ...  100]...[ 153 ... 108  80  50]]
    return b_dec,g_dec,r_dec

def xor_operation_decrypt(b,g,r,mk):
    m,n = b.shape
    bx=np.chararray((m,n))
    gx=np.chararray((m,n))
    rx=np.chararray((m,n))
    b=b.astype(str)
    g=g.astype(str)
    r=r.astype(str)
    for i in range(0,m):
        for j in range (0,n):
            bx[i,j] = dna["{0}{1}".format(b[i,j],mk[i,j])]
            gx[i,j] = dna["{0}{1}".format(g[i,j],mk[i,j])]
            rx[i,j] = dna["{0}{1}".format(r[i,j],mk[i,j])]
         
    bx=bx.astype(str)
    gx=gx.astype(str)
    rx=rx.astype(str)
    return bx,gx,rx 

# read the image and then replace the bits with the new encrypted ones
# saves the image as "enc_<image>.jpg"
def recover_image(b,g,r,iname):
    img = cv2.imread(iname)
    # imread gives BGR format
    # read the image and then replace the bits with the new encrypted ones
    img[:,:,2] = r
    img[:,:,1] = g
    img[:,:,0] = b
    cv2.imwrite(("encrypt_"+iname[:-4]+".jpg"), img)
    print("saved ecrypted image as encrypt_"+iname[:-4]+".jpg")
    return img

# decrypt the entire image and get back original one
def decrypt(image,fx,fy,fz,file_path,Mk):

    print('in dna2, image:', image)
    # split encrypted image into R,G,B 
    r,g,b=split_into_rgb_channels(image)

    p,q = r.shape
    benc,genc,renc=dna_encode(b,g,r)
    bs,gs,rs=scramble_decrypt(fx,fy,fz,benc,genc,renc)
    bx,gx,rx=xor_operation_decrypt(bs,gs,rs,Mk)
    blue,green,red=dna_decode(bx,gx,rx)

    # new image make
    img=np.zeros((p,q,3),dtype=np.uint8)
    img[:,:,0] = red
    img[:,:,1] = green
    img[:,:,2] = blue
    # save the file
    cv2.imwrite(("recovered_"+file_path[:-4]+".jpg"), img)
    
if (__name__ == "__main__"):

    #! user input stuff
    # sends the path to image
    file_path = image_input_from_user()
    print(file_path)
    key = key_input_from_user()



    #! disintegrate image
    
    # image converted to three matrices of R, G, B colors
    blue,green,red=decompose_matrix(file_path)

    #! key encoding stuff

    # m - width, n -height
    key,m,n = securekey(file_path, key)

    # encode key matrix using chaos of blue matrix, DNA encoding -> @returns DNA encoded key matrix
    Mk_e = key_matrix_encode(key,blue)

    #! Lorrenz stuff
    # to generate the x0, y0, and z0 randomly using key
    update_lorenz(key)

    # generates the chaotic Lorenz graph and plots
    x,y,z=gen_chaos_seq(m,n)
    # plot(x,y,z)

    # fx[i] holds the index of where x[i] belongs in the sorted order of x
    fx,fy,fz=sequence_indexing(x,y,z)

    #! Now all operations in R,G,B matrices

    #* 1-> convert red,blue,green matrices decimals -> bits -> DNA encoded letters @returns DNA encoded R, G,B matrices
    blue_e,green_e,red_e=dna_encode(blue,green,red)

    #* 3-> xor the color matrices with key matrix
    blue_final, green_final, red_final = xor_operation_encrypt(blue_e, green_e, red_e, Mk_e)

    
    #* 2-> scramble the blue, green, red arrays wrt fx,fy and fz
    blue_scrambled,green_scrambled,red_scrambled = scramble_encrypt(fx,fy,fz,blue_final,red_final,green_final)
    
    #* 4-> get DNA letters A,C,G,T back to bits (so image original size maintianed)
    b,g,r=dna_decode(blue_scrambled,green_scrambled,red_scrambled)

    # read the image and then replace the bits with the new encrypted ones
    # saves the image as "enc_<image>.jpg"
    encrypted_img=recover_image(b,g,r,file_path)

    #! NOW IMAGE IS ENCRYPTED!!

    print("decrypting...")
    decrypt(encrypted_img,fx,fy,fz,file_path,Mk_e)