# https://github.com/tafseerahmed/image-Encryption-dna-encoding/blob/master/encr.py
# Aim: create a image encryption program using Dna encoding and chaos map

import cv2
import numpy as np
import sys

sys.path.append('../stegano/')
# steg
from stegano import lsb

print("imported")
# import helpers
from dna_helpers import dna_encode, dna_decode, key_matrix_dna_encode, xor_operation_encrypt
from image_helpers import split_into_rgb_channels, decompose_matrix
from user_input_helpers import key_input_from_user, image_input_from_user
from lorenz_helpers import plot, gen_chaos_seq, update_lorenz, sequence_indexing
from key_helpers import securekey
from rsa_helpers import encrypt_using_rsa, decrypt_using_rsa


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

# read the image and then replace the bits with the new encrypted ones
# saves the image as "enc_<image>.jpg"
def recover_image(b,g,r,iname):
    img = cv2.imread(iname)
    # imread gives BGR format
    # read the image and then replace the bits with the new encrypted ones
    img[:,:,2] = r
    img[:,:,1] = g
    img[:,:,0] = b
    cv2.imwrite(("encrypt_"+iname[:-4]+".png"), img)
    print("saved ecrypted image as encrypt_"+iname[:-4]+".png")

    # readed = cv2.imread("encrypt_"+iname[:-4]+".png")
    # print('readed', readed)
    return img
    
if (__name__ == "__main__"):

    #! user input stuff
    # sends the path to image
    file_path = image_input_from_user()
    print(file_path)
    key = key_input_from_user()

    #!rsa
    rsa_encrypted_key = encrypt_using_rsa(key)
    print("rsa_encrypted_key", len(rsa_encrypted_key))

    # utf_rsa_crypto = rsa_encrypted_key.decode('cp1251').encode('utf8')

    
    
    # try decrypting rsa
    # rsa_decrypted = decrypt_using_rsa(rsa_encrypted_key)
    # print("decrypted from crypto", rsa_decrypted)

    #! steganography
    secret = lsb.hide("./lena.png", rsa_encrypted_key)
    secret.save("./lena_stego.png")
    print("stego lena saved!")
    stego_revealed_message = lsb.reveal("./lena_stego.png")
    print("stego rsa_encrypted_key", len(stego_revealed_message))
    
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
    # print('in encryption', encrypted_img.shape)