#!/usr/in/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:02:30 2023

@author: christopherrichardson
"""

# In[]

import glob, os
import pandas as pd

from PIL import Image
import matplotlib.pyplot as plt



def renamer(new_file_name, ok=1):
    # remove spaces
    nfn = new_file_name.replace(' ','').upper()

    # check if length is 25
    if len(nfn) == 25:
        pass
    else:
        raise ValueError('Char count is off!!!')

    # if image is mehhh, add prefix ok_
    if ok == 1:
        pass
    elif ok == 2:
        nfn = 'ok_' + nfn
    elif ok == 3:
        nfn = 'n_' + nfn
    
    return nfn

# In[]

path = '/Users/christopherrichardson/Library/CloudStorage/GoogleDrive-christopherr@sandiego.edu/My Drive/Alpha Projekt Scan/n/'

os.chdir(path)
files = os.listdir(path)

# In[]

image_paths = []
# for file in glob.glob(path + '*.jpg'):
#     print(file)
#     images.append(file)

for file in files:
    if file[-4:] == '.jpg':
        # print(file)
        image_paths.append(file)

# In[ print file ]
num = 1
current_file = image_paths[num]
print(current_file)
 # In[ view image ]
scope = Image.open(current_file)
plt.imshow(scope)

# In[]
# bool_ = 1 # great
bool_ = 2 # ok
# bool_ = 3 # meh

# In[ type in keys from image ]
 # CURRENTLY @ 23
keys = 'bncm7 gkyJH 6CJKV YKR3F 3DCY7'

# In[ clean up string & display]
keys = renamer(keys, ok=bool_)

print(keys)
# print(current_file)
# In[]
os.rename(current_file, keys + '.jpg')
print(f'{keys} saved!')





# In[]
len(image_paths)




# In[]



# In[]




# In[]





# In[]





# In[]



# In[]




# In[]





# In[]





# In[]



# In[]




# In[]





# In[]



