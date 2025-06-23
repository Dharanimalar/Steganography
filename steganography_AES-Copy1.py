#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install opencv-python pycryptodome matplotlib numpy')


# In[2]:


import cv2
import hashlib
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display


# In[3]:


# SHA-256 key derivation
def derive_key(userkey):
    return hashlib.sha256(userkey.encode()).digest()[:16]

# AES Encryption (returns IV + ciphertext)
def encrypt_message(msg, userkey):
    key = derive_key(userkey)
    cipher = AES.new(key, AES.MODE_CBC)
    ct = cipher.encrypt(pad(msg.encode(), AES.block_size))
    return cipher.iv + ct

# AES Decryption
def decrypt_message(cipher_bytes, userkey):
    key = derive_key(userkey)
    iv = cipher_bytes[:16]
    ct = cipher_bytes[16:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(ct), AES.block_size).decode()


# In[4]:


image_path = r"C:\Users\sindh\Downloads\0ab09927a3e9f81653edfab3297c8cbc.jpg" 
img = cv2.imread(image_path)

if img is None:
    print("Image not found.")
else:
    print("✅ Image loaded:", img.shape)


# In[5]:


key = "123"
text = "This is a top secret message!"

# Encrypt message
encrypted_bytes = encrypt_message(text, key)
length = len(encrypted_bytes)

# Show encrypted object
print("🔐 Encrypted Object (hex):")
print(encrypted_bytes.hex())

# Embed length (2 bytes)
len_bytes = length.to_bytes(2, byteorder='big')
final_data = len_bytes + encrypted_bytes

# XOR with key and embed
d = {chr(i): i for i in range(256)}
n, m, z = 0, 0, 0
k1 = 0

stego_img = img.copy()

for i in range(len(final_data)):
    stego_img[n, m, z] = final_data[i] ^ d[key[k1]]
    n += 1
    if n >= img.shape[0]:
        n = 0
        m += 1
    z = (z + 1) % 3
    k1 = (k1 + 1) % len(key)

cv2.imwrite("stego_output.png", stego_img)
print("✅ Message embedded and saved as 'stego_output.png'")


# In[6]:


# Convert BGR to RGB for matplotlib
original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
stego_rgb = cv2.cvtColor(stego_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(original_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(stego_rgb)
plt.title("Stego Image")
plt.axis("off")
plt.tight_layout()
plt.show()


# In[7]:


key1 = input("🔑 Re-enter key to extract message: ")

if key1 == key:
    x = cv2.imread("stego_output.png")
    d = {chr(i): i for i in range(256)}
    n, m, z = 0, 0, 0
    k1 = 0

    # Extract length
    length_bytes = bytearray()
    for _ in range(2):
        length_bytes.append(x[n, m, z] ^ d[key[k1]])
        n += 1
        if n >= x.shape[0]:
            n = 0
            m += 1
        z = (z + 1) % 3
        k1 = (k1 + 1) % len(key)

    extracted_len = int.from_bytes(length_bytes, byteorder='big')

    # Extract encrypted data
    encrypted_back = bytearray()
    for _ in range(extracted_len):
        encrypted_back.append(x[n, m, z] ^ d[key[k1]])
        n += 1
        if n >= x.shape[0]:
            n = 0
            m += 1
        z = (z + 1) % 3
        k1 = (k1 + 1) % len(key)

    try:
        decrypted_text = decrypt_message(encrypted_back, key1)
        print("✅ Decrypted Message:", decrypted_text)
    except Exception as e:
        print("❌ Decryption failed:", e)
else:
    print("❌ Key mismatch. Access denied.")


# In[ ]:




