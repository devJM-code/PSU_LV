import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = plt.imread("tiger.png")
img = img[:,:,0].copy()

print(img.shape)
print(img.dtype)

img = np.matrix_transpose(img)
img = np.flip(img, axis = 1)
img = np.flip(img, axis = 0)

img = img[::10, ::10]

x_max = img.shape[0]
y_max = img.shape[1]

y_quarter = int(y_max * 0.25)

img[ : , : y_quarter] = 0x000000
img[ : , 2 * y_quarter : ] = 0x000000

max_v = img.max()

plt.figure()
plt.imshow(img, cmap = "gray", vmax = max_v * 0.85)
plt.show()
