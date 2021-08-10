import cv2
import seaborn as sns
import matplotlib.pyplot as plt



sns.set(color_codes=True)
# Read the image
image = cv2.imread('Figure_1.png') #--imread() helps in loading an image into jupyter including its pixel values

# as opencv loads in BGR format by default, we want to show it in RGB.
#plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.show()