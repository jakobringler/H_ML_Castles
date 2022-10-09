# code for displaying multiple images in one figure
  
#import libraries
import cv2
from matplotlib import pyplot as plt
  
# create figure
fig = plt.figure(figsize=(20, 5))

# setting values to rows and column variables
rows = 1
columns = 4

def shuffle(Image):
    image = Image
    (R, G, B) = cv2.split(image)
    merged = cv2.merge([B, G, R])
    return merged
  
# reading images
Image1 = cv2.imread('tower_clean.png')
Image2 = cv2.imread('tower_lowFreq.png')
Image3 = cv2.imread('tower_highFreq.png')
Image4 = cv2.imread('tower_transform.png')

# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)
  
# showing image
plt.imshow(shuffle(Image1))
plt.axis('on')
plt.title("Clean")
  
# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)
  
# showing image
plt.imshow(shuffle(Image2))
plt.axis('on')
plt.title("Low Freq Noise")
  
# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 3)
  
# showing image
plt.imshow(shuffle(Image3))
plt.axis('on')
plt.title("High Freq Noise")
  
# Adds a subplot at the 4th position
fig.add_subplot(rows, columns, 4)
  

# showing image
plt.imshow(shuffle(Image4))
plt.axis('on')
plt.title("Transformed")

fig.tight_layout(w_pad=5)   

plt.show()