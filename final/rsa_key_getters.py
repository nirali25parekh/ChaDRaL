import rsa

def get_A_public_key():
    A_pub_file = open('A_public_key.txt', mode='rb')
    keydata = A_pub_file.read()
    actualkey = rsa.PublicKey.load_pkcs1(keydata)
    print(actualkey)
    return actualkey

def get_B_public_key():
    B_pub_file = open('B_public_key.txt', mode='rb')
    keydata = B_pub_file.read()
    actualkey = rsa.PublicKey.load_pkcs1(keydata)
    return actualkey

def get_A_private_key():
    A_priv_file = open('A_private_key.txt', mode='rb')
    keydata = A_priv_file.read()
    actualkey = rsa.PrivateKey.load_pkcs1(keydata)
    return actualkey
    
def get_B_private_key():
    B_priv_file = open('B_private_key.txt', mode='rb')
    keydata = B_priv_file.read()
    actualkey = rsa.PrivateKey.load_pkcs1(keydata)
    return actualkey
