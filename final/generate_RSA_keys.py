import rsa

# A's keys
(A_pub, A_priv) = rsa.newkeys(128)

A_pub_file = open("A_public_key.txt","w")
A_pub_file.write(A_pub.save_pkcs1().decode('utf-8'))
A_pub_file.close()

A_priv_file = open("A_private_key.txt","w")
A_priv_file.write(A_priv.save_pkcs1().decode('utf-8'))
A_priv_file.close()

print("A's public and private key saved in file")

# B's keys
(B_pub, B_priv) = rsa.newkeys(512)

B_pub_file = open("B_public_key.txt","w")
B_pub_file.write(B_pub.save_pkcs1().decode('utf-8'))
B_pub_file.close()

B_priv_file = open("B_private_key.txt","w")
B_priv_file.write(B_priv.save_pkcs1().decode('utf-8'))
B_priv_file.close()

print("B's public and private key saved in file")
