# https://medium.com/quick-code/aes-implementation-in-python-a82f582f51c2

import hashlib
from Crypto import Random
from Crypto.Cipher import AES
from base64 import b64encode, b64decode



# AES block_size is set to 128
# constructor receives a key, we generate a 256 bit hash from that key
def __init__(self, key):
    self.block_size = AES.block_size
    self.key = hashlib.sha256(key.encode()).digest()

# receives plain text, and adds the number of bytes for it ot be a multiple of 128
def __pad(self, plain_text):
    number_of_bytes_to_pad = self.block_size - len(plain_text) % self.block_size
    # ascii_string is our padding character -> no of extra padding number's ascii value
    ascii_string = chr(number_of_bytes_to_pad)
    # padding_str is our character that many number of times
    padding_str = number_of_bytes_to_pad * ascii_string
    # padded with the padding character
    padded_plain_text = plain_text + padding_str
    return padded_plain_text

@staticmethod
def __unpad(plain_text):
    last_character = plain_text[len(plain_text) - 1:]
    # last_character is the ascii value of the number of padding places
    bytes_to_remove = ord(last_character)
    return plain_text[:-bytes_to_remove]

def encrypt(self, plain_text):
    # 1. plain text -> padding
    plain_text = self.__pad(plain_text)
    # 2. generate random iv with size of block i.e. 128bits
    iv = Random.new().read(self.block_size)
    # 3. generate our cipher with our key and iv
    cipher = AES.new(self.key, AES.MODE_CBC, iv)
    # 4. encrypt our text        
    encrypted_text = cipher.encrypt(plain_text.encode())
    # 5. bits -> readable characters
    return b64encode(iv + encrypted_text).decode("utf-8")


def decrypt(self, encrypted_text):
    # readable -> bits
    encrypted_text = b64decode(encrypted_text)
    # extract the iv(1st 128 bits of our encrypted text)
    iv = encrypted_text[:self.block_size]
    # generate our cipher with our key and iv
    cipher = AES.new(self.key, AES.MODE_CBC, iv)
    # decrypt using our cipher
    plain_text = cipher.decrypt(encrypted_text[self.block_size:]).decode("utf-8")
    # unpad 
    return self.__unpad(plain_text)

encrypt_image_path = 'image_encrypt.jpg'

try:
	# take path of image as a input
	path = input(r'Enter path of Image : ')
	
	# taking decryption key as input
	key = int(input('Enter Key for encryption of Image : '))

    # open file for reading purpose
	fin = open(path, 'rb')
	
	# storing image data in variable "image"
	image = fin.read()
	fin.close()
	
	# converting image into byte array to
	# perform encryption easily on numeric data
	image = bytearray(image)

    image = encrypt(image)

    # opening file for writing purpose
    fin = open(encrypt_image_path, 'wb')
	
	# writing encrypted data in image
	fin.write(image)
	fin.close()
	print('Encryption Done...')

except Exception:
	print('Error caught : ', Exception.__name__)
