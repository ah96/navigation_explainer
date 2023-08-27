#!/usr/bin/env python3

import numpy as np
import copy
from matplotlib import pyplot as plt
from PIL import Image

explanation_size_y = 600
explanation_size_x = 600

image = Image.open('explanation.png')
explanation = np.asarray(image)

fig = plt.figure(frameon=True)
w = 0.01 * explanation_size_x
h = 0.01 * explanation_size_y
fig.set_size_inches(w, h)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(explanation) #np.flip(explanation))#.astype(np.uint8))

ax.text(420, 35, 'door', c='white', fontsize=11, fontweight=20.0)

ax.text(370, 150, 'human', c='white', fontsize=17, fontweight=20.0)

# CONVERT IMAGE TO NUMPY ARRAY 
fig.savefig('tiago_library_exp' + '.png', transparent=False)
plt.close()
