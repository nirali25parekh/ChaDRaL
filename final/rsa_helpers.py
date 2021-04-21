import rsa
from rsa_key_getters import get_B_public_key, get_B_private_key

def encrypt_using_rsa(message):
    message_bin = message.encode('utf8')
    B_pub_key = get_B_public_key()
    crypto = rsa.encrypt(message_bin, B_pub_key)
    return crypto

def decrypt_using_rsa(crypto):

    B_priv_key = get_B_private_key()
    message_bin = rsa.decrypt(crypto, B_priv_key)
    decoded = message_bin.decode('utf8')
    return decoded
