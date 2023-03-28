#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 11:32:10 2023

@author: christopherrichardson

"""
# In[Img Load]
image_path = '/Users/christopherrichardson/Library/CloudStorage/GoogleDrive-christopherr@sandiego.edu/My Drive/_Projekt_Skan/COA_imgs/demo.jpg

# load and convert to grayscale (2nd param == 0)
og_img = cv.imread(image_path)
og_img = cv.medianBlur(og_img,5)
show(og_img)

 # In[alignment DEL]
 # X

# ---- TO DELTE 
# rotate img to near true vertical
img_center = tuple(np.array(og_img.shape[1::-1]) / 2)
rot_mat = cv.getRotationMatrix2D(img_center, .8, 1.0)
og_img = cv.warpAffine(og_img, rot_mat, og_img.shape[1::-1], flags=cv.INTER_LINEAR)

show(og_img)

# In[ isolate yellow background ]
# REF https://stackoverflow.com/questions/10948589/choosing-the-correct-upper-and-lower-hsv-boundaries-for-color-detection-withcv

# lower and uppper limits of the yellow background
lower = np.array([15, 50, 200])
upper = np.array([35, 200, 255])

imgHSV = cv.cvtColor(og_img, cv.COLOR_BGR2HSV)
mask = cv.inRange(imgHSV, lower, upper)
# inverse mask
mask = 255-mask
og_img = cv.bitwise_not(og_img, og_img, mask=mask)

show(og_img)

 # In[ whole img threshold/outline to DF ]
 # REF https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html
 
# eroding @ window_size 11 actually dilates all highlighted parts of the img from the yellow mask above
# removes all small imperfections, making it an alternative to blur with an emphasis on yellow
window_size =11
kernel = np.ones((window_size,window_size))
mask = cv.erode(mask,kernel,iterations = 2)

# In[ find outline of label ]

# # In Search for bottom left Square of Windows logof
# ret, thresh = cv.threshold(mask, 127, 255, 0)
(og_h, og_w) = mask.shape[:2]
image_size = og_h * og_w
mser = cv.MSER_create()
mser.setMaxArea(round(image_size/2))
mser.setMinArea(2)

_, bw = cv.threshold(mask, 0.0, 255.0, cv.THRESH_OTSU )
regions, rects = mser.detectRegions(bw)
df = pd.DataFrame(rects, columns=['x','y','w','h'])

 # In[ filter out all outlines but Windows Logo ]

# filters out everything but lower left square of the Windows logo
# this would be calibrated dependent on pixel density of scan
width_bool = (df['w'] > 100) & (df['w'] <200)
height_bool = (df['h'] > 100) & (df['h'] < 200)

df = df.loc[width_bool & height_bool,]

df.sort_values(by='y', inplace=True)
df.reset_index(drop=True, inplace=True)

og_h, og_w = mask.shape[:2]
                  #  y , x
canvas = np.zeros((og_h,og_w), dtype='uint8')

texts = []

for idx, row in df.iterrows():
    x, y, w, h = [row['x'], row['y'], row['w'], row['h']]

    cv.rectangle(canvas, (x, y), (x+w, y+h), color=(255, 255, 200), thickness=15)
    
    text = return_chars(og_img, x,y,w,h)
    texts.append(text)

 # In[]

show(canvas)
 # In[]

show(texts[0])

 # In[]
mask = texts[0]

# convert mask to 3 channel
# for padding manipulation
mask = np.tile(mask[:, :, None], [1, 1, 3])
img_ = add_padding(mask, padding=50, r=255, g=255, b=255)
# convert back to single channel
gray = cv.cvtColor(img_, cv.COLOR_BGR2GRAY)

# In[ find all chars ]
# Finds all shapes, in our case chars 


(og_h, og_w) = gray.shape[:2]
image_size = og_h * og_w
mser = cv.MSER_create()
mser.setMaxArea(round(image_size/2))
mser.setMinArea(2)

_, bw = cv.threshold(gray, 0.0, 255.0, cv.THRESH_OTSU )
regions, rects = mser.detectRegions(bw)

 # In[ Remove all bounding boxes found within Chars & minor specs ]
 # i.e. B, D, O, P, Q, R all have inside shapes that will be detected
 # most minor specs are fragments of debris resulted from scratching COAs

df = pd.DataFrame(rects, columns=['x','y','w','h'])
# remove all small imperfections and small boxes from characters like B,8, P, Q, O, etc
df = df.loc[(df['w'] > 15) & (df['h'] > 25),].reset_index(drop=True)

idxs_2_delete = []
for c_idx, c_row in df.iterrows():
    '''
        The following checks every row vs all other rows (boxes) to see if the starting point
        is within another bounding box (row).
    '''
    cx, cy, cw, ch = [c_row['x'], c_row['y'], c_row['w'], c_row['h']]

    sum_ = 0
    for idx, row in df.iterrows():
        x, y, w, h = [row['x'], row['y'], row['w'], row['h']]

        if idx != c_idx:
            # ** check if current box is indeed within another box
            if x < cx < x+w and y < cy < y+h:
                sum_ += 1

    idxs_2_delete.append(sum_)

# drop boxes that are marked 1 based on bool
df['bool'] = idxs_2_delete
df = df.loc[df['bool'] != 1,:]
df.drop(columns=['bool'], inplace=True)
df.reset_index(drop=True, inplace=True)

 # In[ allign all text via Y value ]
 # alligning text via Y value allows us to sort the 1st char to the last,
 # where the program starts at the highest Y, then sorts left to right (X),
 # thus each row is alligned


df = pd.DataFrame(rects, columns=['x','y','w','h'])
# remove all small imperfections and small boxes from characters like B,8, P, Q, O, etc
df = df.loc[(df['w'] > 15) & (df['h'] > 25),].reset_index(drop=True)

idxs_2_delete = []
for c_idx, c_row in df.iterrows():
    '''
        The following checks every row vs all other rows (boxes) to see if the starting point
        is within another bounding box (row).
    '''
    cx, cy, cw, ch = [c_row['x'], c_row['y'], c_row['w'], c_row['h']]

    sum_ = 0
    for idx, row in df.iterrows():
        x, y, w, h = [row['x'], row['y'], row['w'], row['h']]

        if idx != c_idx:
            # ** check if current box is indeed within another box
            if x < cx < x+w and y < cy < y+h:
                sum_ += 1

    idxs_2_delete.append(sum_)

df['bool'] = idxs_2_delete

# drop boxes that are marked 1
df = df.loc[df['bool'] != 1,:]
df.drop(columns=['bool'], inplace=True)
df.reset_index(drop=True, inplace=True)

# !!!! 
# since there are 3 rows, we should only have to go through the df 3 times
# !!!! 

# 1st group
y = df.loc[0,'y']

df1 = df.loc[(df['y'] > y-20) & (df['y'] < y+20), :]
df = df.drop(df1.index).reset_index(drop=True)

df1['y'] = round(df1['y'].mean())


y = df.loc[0,'y']

df2 = df.loc[(df['y'] > y-20) & (df['y'] < y+20), :]
df = df.drop(df2.index).reset_index(drop=True)

df2['y'] = round(df2['y'].mean())

y = df.loc[0,'y']

df3 = df.loc[(df['y'] > y-20) & (df['y'] < y+20), :]
df3.loc[:,'y'] = round(df3['y'].mean())

df = pd.concat([df1, df2, df3])
df.reset_index(drop=True, inplace=True)

df.sort_values(by=['y','x'], inplace=True)
df.reset_index(drop=True, inplace=True)

 # In[!!! _____ exp]


canvas = np.ones((200,1400,3), np.uint8)*255
canvas = cv.cvtColor(canvas, cv.COLOR_BGR2GRAY)

chars = []
new_xs = []

for idx, row in df.iterrows():
    # the following only uses 1st row X as a starting point and disregards all other starting points
    
    # use the same h & y for all blocks
    new_y = df['y'].min()

    # if 1st row
    if idx == 0:
        c_x, c_y, c_w, c_h = [row['x'], row['y'], row['w'], row['h']]
        new_x = c_x.copy()
        new_xs.append(new_x)
        
    else:
        # we create our our new X based on last row's ending X + W + 20 (padding)
        # to get current row's starting point
        last_char_end = new_xs[idx-1] + df.loc[idx-1,'w'] + 20
        # current row calculated starting point
        new_x = last_char_end.copy()
        new_xs.append(new_x)
        
        # curernt char x, y, w, h
        c_x, c_y, c_w, c_h = [row['x'], row['y'], row['w'], row['h']]
        
    char = gray[c_y:c_y + c_h, c_x:c_x + c_w]
    char = add_padding(char,padding=15)
    # -next 3 lines 4 debugging 
    # chars.append(char)
    # show(char)
    # plt.show()
    
    char = cv.cvtColor(char, cv.COLOR_BGR2GRAY)
    canvas[new_y:new_y + c_h + 15 , new_x:new_x + c_w + 15] = char.copy()
        
show(canvas)   

# In[]

cv.imwrite('label1.jpg', canvas)



