import numpy as np
import cv2

# import helpers
from dna_helpers import dna_encode, dna_decode, key_matrix_dna_encode, xor_operation_decrypt
from image_helpers import split_into_rgb_channels, decompose_matrix
from user_input_helpers import key_input_from_user, image_input_from_user
from key_helpers import securekey
from lorenz_helpers import gen_chaos_seq, update_lorenz, sequence_indexing
from rsa_helpers import decrypt_using_rsa

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

# decrypt the entire image and get back original one
def decrypt(fx,fy,fz,file_path,Mk):
    image = cv2.imread(file_path)
#     print('in decryption, image:', image)
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

    # crypto = input("enter crypto")
    # print(type(crypto))
    # crypto = bytes(crypto, 'utf-8')
    # key = decrypt_using_rsa(crypto)
    # print("in decryption, key is", key)
    file_path = image_input_from_user()
    key = key_input_from_user()

    #! disintegrate image
    
    # image converted to three matrices of R, G, B colors
    blue,green,red=decompose_matrix(file_path)

    #! key encoding stuff

    # m - width, n -height
    key,m,n = securekey(file_path, key)
    
    # encode key matrix using chaos of blue matrix, DNA encoding -> @returns DNA encoded key matrix
    Mk_e = key_matrix_dna_encode(key,blue)

    #! Lorrenz stuff
    # to generate the x0, y0, and z0 randomly using key
    update_lorenz(key)

    # generates the chaotic Lorenz graph and plots
    x,y,z=gen_chaos_seq(m,n)
    # plot(x,y,z)

    # fx[i] holds the index of where x[i] belongs in the sorted order of x
    fx,fy,fz=sequence_indexing(x,y,z)

    #! decryption

    print("decrypting...")
    decrypt(fx,fy,fz,file_path,Mk_e)