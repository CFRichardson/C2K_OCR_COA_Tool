#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 11:32:10 2023

@author: christopherrichardson


2 PARAMETERS FOR WHOLE SYSTEM!!!

  Parameters
  ----------
  file_location : jpeg file
      Scans of COAs edged against scanner
  number of strips : integer
      number of strips in jpeg/scan

  Returns
  -------
  csv file with COAs

"""

# In[Img Load]
train_dir = '/Users/christopherrichardson/Library/CloudStorage/GoogleDrive-christopherr@sandiego.edu/My Drive/Alpha Projekt Scan/train_imgs'


# file_ = 'tri_1'
# file_ = 'tri_2'
# file_ = 'tri_3'
file_ = 'tri_4'
# file_ = 'tri_5'
# file_ = 'tri_6'
# file_ = 'mono_1'
# file_ = 'mono_2' 

image_path = f'/Users/christopherrichardson/Library/CloudStorage/GoogleDrive-christopherr@sandiego.edu/My Drive/Alpha Projekt Scan/COA_imgs/{file_}.jpg'


# load and convert to grayscale (2nd param == 0)
og_img = cv.imread(image_path)
# og_img = cv.medianBlur(og_img,5)
show(og_img)


# In[]

# DELETE
first_label = og_img[300:500, 500:1000].copy()
show(first_label)


# In[ isolate yellow background ]

# lower and uppper limits of the yellow background
lower = np.array([15, 50, 200])
upper = np.array([35, 200, 255])

imgHSV = cv.cvtColor(og_img, cv.COLOR_BGR2HSV)
mask = cv.inRange(imgHSV, lower, upper)
# inverse mask
mask = 255-mask
# og_img = cv.bitwise_not(og_img, og_img, mask=mask)

show(mask)


# In[ display erosion ]

# erosion of mask (really dilation) improves highlights for mser.detectRegions
window_size =11
kernel = np.ones((window_size,window_size))
mask = cv.erode(mask,kernel,iterations=1)

show(mask)
# In[]


# In Search for bottom left Square of Windows logof
ret, thresh = cv.threshold(mask, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

(og_h, og_w) = mask.shape[:2]
image_size = og_h * og_w
mser = cv.MSER_create()
mser.setMaxArea(round(image_size/2))
mser.setMinArea(2)

_, bw = cv.threshold(mask, 0.0, 255.0, cv.THRESH_OTSU )

regions, rects = mser.detectRegions(bw)

df = pd.DataFrame(rects, columns=['x','y','w','h'])


 # In[ filter out all outlines but Windows Logo ]
 # return_chars currently only works with single strip
 
 # ********* FINAL VERSION NEEDS TO SHOW THIS BLOCK AS AN OUTPUT
 # NEEDED FOR MIDDLE STRIP
 # IF IRREGULARITIES APPEAR, OUTLIERS SHOULD BE REMOVED VIA HUMAN INTERVENTION

# # filters out everything but lower left square of the Windows logo
# # this would be calibrated dependent on pixel density of scan
width_bool = (df['w'] > 100) & (df['w'] <200)
height_bool = (df['h'] > 100) & (df['h'] < 200)
df = df.loc[width_bool & height_bool,]

# organize strips by X-axis to find 1st strip
df.sort_values(by='x', inplace=True)
df.reset_index(drop=True, inplace=True)

# canvas creation
og_h, og_w = mask.shape[:2]
                  #  y , x
canvas = np.zeros((og_h,og_w), dtype='uint8')

# plot window logo shapes found
texts = []
for idx, row in df.iterrows():
    x, y, w, h = [row['x'], row['y'], row['w'], row['h']]
    cv.rectangle(canvas, (x, y), (x+w, y+h), color=(255, 255, 200), thickness=15)
show(canvas)

# In[ 1st strip ]
### POSSIBLE TO MAKE THE FOLLOWING STEPS INTO AN APPLY FUNCTION

og_h, og_w = mask.shape[:2]
                  #  y , x
canvas = np.zeros((og_h,og_w), dtype='uint8')

strip1_texts = []
for idx, row in df.iterrows():
    
    x, y, w, h = [row['x'], row['y'], row['w'], row['h']]
    if x < 300:

        cv.rectangle(canvas, (x, y), (x+w, y+h), color=(255, 255, 200), thickness=15)
        
        text = img_scope(og_img,x,y,w,h, x_s=400, x_e=750, y_s=65, y_e=125,bs=(350,150), be=(500,200))
        strip1_texts.append(text)
show(canvas)

# In[]

for num_, img_ in enumerate(strip1_texts):
    show(img_)
    plt.show()
        
# In[]

for num_, img_ in enumerate(strip1_texts):
 
    cv.imwrite(f'{train_dir}/{file_}_strip1_label_{num_}.jpg', img_)

# In[ 2nd strip ]
# 
# 2nd strip has problems, possibly due to scanner light being the brightest in the middle, thus
# unwanted squares are sometimes shown, here we take the the 2nd 

bool_ = (df['x'] > 1500) & (df['x'] < 2500)
df_2nd_strip = df.loc[bool_,:]

 # ********* FINAL VERSION HAS THIS AS A BOOL FOR END USER BASED ON IF
 # MIDDLE STRIP IRREGULARITIES APPEAR, OUTLIERS SHOULD BE REMOVED VIA HUMAN INTERVENTION
# remove any outliers
df_2nd_strip = df_2nd_strip.loc[(np.abs(stats.zscore(df_2nd_strip['x'])) < 1),:]

og_h, og_w = mask.shape[:2]
                  #  y , x
canvas = np.zeros((og_h,og_w), dtype='uint8')

strip2_texts = []
for idx, row in df_2nd_strip.iterrows():
    x, y, w, h = [row['x'], row['y'], row['w'], row['h']]

    cv.rectangle(canvas, (x, y), (x+w, y+h), color=(255, 255, 200), thickness=15)
    text = img_scope(og_img, x,y,w,h, x_s=270, x_e=590, y_s=210, y_e=275,bs=(350,150), be=(500,200))
    strip2_texts.append(text)

show(canvas)

# In[]

for num_, img_ in enumerate(strip2_texts):
    show(img_)
    plt.show()
    
# In[]

for num_, img_ in enumerate(strip2_texts):    
    cv.imwrite(f'{train_dir}/{file_}_strip2_label_{num_}.jpg', img_)

 # In[ 3rd strip ]

bool_ = (df['x'] > 2500)
df_3rd_strip = df.loc[bool_,:]


og_h, og_w = mask.shape[:2]
                  #  y , x
canvas = np.zeros((og_h,og_w), dtype='uint8')

strip3_texts = []
for idx, row in df_3rd_strip.iterrows():
    x, y, w, h = [row['x'], row['y'], row['w'], row['h']]

    cv.rectangle(canvas, (x, y), (x+w, y+h), color=(255, 255, 200), thickness=15)
    text = img_scope(og_img, x,y,w,h, x_s=270, x_e=590, y_s=210, y_e=270,bs=(350,150), be=(500,200))
    strip3_texts.append(text)
show(canvas)

# In[]

for num_, img_ in enumerate(strip3_texts):
    show(img_)
    plt.show()
    
# In[]

for num_, img_ in enumerate(strip3_texts):
    cv.imwrite(f'{train_dir}/{file_}_strip3_label_{num_}.jpg', img_)


# In[]




 # In[]






 # In[]



 # In[]






 # In[]



 # In[]






 # In[]



 # In[]


