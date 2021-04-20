from PIL import Image
import hashlib 

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
    key = hashlib.sha256(key_in_bytes)                        

    # image data is fed to generate digest
    # key.update(bytearray(plainimage)) 

    # hexdigest() : Returns the encoded data in hexadecimal format -> length 64
    key_hex_digest = key.hexdigest()
    #// key_hex_digest: 331332a.....0b0670
    return  key_hex_digest, m, n
