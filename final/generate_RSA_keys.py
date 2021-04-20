import rsa

(A_pub, A_priv) = rsa.newkeys(512)
print("A's public key and private key generated")

(B_pub, B_priv) = rsa.newkeys(512)
print("B's public and private key generated")

def get_A_public_key:
    return A_pub

def get_B_public_key:
    return B_pub

def get_A_private_key:
    return A_priv
    
def get_B_private_key:
    return B_priv
