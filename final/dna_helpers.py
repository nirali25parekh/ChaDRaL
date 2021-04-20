import numpy as np

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

    print(b_dec.shape)
    #// b_dec = [[1 1 0 ... 1 0 0] ...[]]
    b_dec=(np.packbits(b_dec,axis=-1))
    g_dec=(np.packbits(g_dec,axis=-1))
    r_dec=(np.packbits(r_dec,axis=-1))
    # print(b_dec)
    print(b_dec.shape)
    #// b_dec = [[198 110 254 ...  100]...[ 153 ... 108  80  50]]
    return b_dec,g_dec,r_dec

# encode key matrix using chaos of b matrix, DNA encoding
def key_matrix_dna_encode(key, b):    
    
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