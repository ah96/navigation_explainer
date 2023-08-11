from PIL import Image
import numpy as np
# 1. Read image
img = Image.open('map_1.png')
 
# 2. Convert image to NumPy array
arr = np.asarray(img)
print(arr.shape)
arr = arr[600:1400,600:1400]
# 3. Convert 3D array to 2D list of lists
lst = []
for row in arr:
    tmp = []
    for col in row:
        tmp.append(str(col))
    lst.append(tmp)
# 4. Save list of lists to CSV
with open('map_1.csv', 'w') as f:
    for row in lst:
        f.write(','.join(row) + '\n')