#!/usr/bin/env python3

import numpy as np
import copy
from matplotlib import pyplot as plt

explanation_size_y = 600
explanation_size_x = 600

table_cx = 280
table_cy = 270
table_dx = 200
table_dy = 100

chair1_cx = 230
chair1_cy = 400
chair1_dx = 80
chair1_dy = 80

chair2_cx = 330
chair2_cy = 320
chair2_dx = 80
chair2_dy = 80

wall1_cx = 300
wall1_cy = 575
wall1_dx = 600
wall1_dy = 50

bookshelf_cx = 480
bookshelf_cy = 480
bookshelf_dx = 200
bookshelf_dy = 100

explanation_R = np.zeros((explanation_size_y, explanation_size_x))
explanation_R[:,:] = 120
explanation_R[int(table_cy-0.5*table_dy):int(table_cy+0.5*table_dy), int(table_cx-0.5*table_dx):int(table_cx+0.5*table_dx)] = 180
explanation_R[int(chair1_cy-0.5*chair1_dy):int(chair1_cy+0.5*chair1_dy), int(chair1_cx-0.5*chair1_dx):int(chair1_cx+0.5*chair1_dx)] = 180
explanation_R[int(chair2_cy-0.5*chair2_dy):int(chair2_cy+0.5*chair2_dy), int(chair2_cx-0.5*chair2_dx):int(chair2_cx+0.5*chair2_dx)] = 180
explanation_R[int(wall1_cy-0.5*wall1_dy):int(wall1_cy+0.5*wall1_dy), int(wall1_cx-0.5*wall1_dx):int(wall1_cx+0.5*wall1_dx)] = 180
explanation_R[int(bookshelf_cy-0.5*bookshelf_dy):int(bookshelf_cy+0.5*bookshelf_dy), int(bookshelf_cx-0.5*bookshelf_dx):int(bookshelf_cx+0.5*bookshelf_dx)] = 180
temp = copy.deepcopy(explanation_R)

explanation_G = copy.deepcopy(explanation_R)
explanation_B = copy.deepcopy(explanation_R)

c_x_pixel = 115 #int(0.5*explanation_size_x)
c_y_pixel = 250 #int(0.5*explanation_size_y)
d_x = 1200
d_y = 1200
#print('(d_x, d_y) = ', (d_x, d_y))

# FIRST
explanation_R[max(0, c_y_pixel-int(0.1*d_y)):min(explanation_size_y, c_y_pixel+int(0.1*d_y)), max(0, c_x_pixel-int(0.1*d_x)):min(explanation_size_x, c_x_pixel+int(0.1*d_x))] = 227
explanation_G[max(0, c_y_pixel-int(0.1*d_y)):min(explanation_size_y, c_y_pixel+int(0.1*d_y)), max(0, c_x_pixel-int(0.1*d_x)):min(explanation_size_x, c_x_pixel+int(0.1*d_x))] = 242
explanation_B[max(0, c_y_pixel-int(0.1*d_y)):min(explanation_size_y, c_y_pixel+int(0.1*d_y)), max(0, c_x_pixel-int(0.1*d_x)):min(explanation_size_x, c_x_pixel+int(0.1*d_x))] = 19


# SECOND
explanation_R[max(0, c_y_pixel-int(0.2*d_y)):min(explanation_size_y, c_y_pixel+int(0.2*d_y)), max(0, c_x_pixel-int(0.2*d_x)):max(0, c_x_pixel-int(0.1*d_x))] = 206
explanation_R[max(0, c_y_pixel-int(0.2*d_y)):min(explanation_size_y, c_y_pixel+int(0.2*d_y)), min(explanation_size_x, c_x_pixel+int(0.1*d_x)):min(explanation_size_x, c_x_pixel+int(0.2*d_x))] = 206
explanation_R[max(0, c_y_pixel-int(0.2*d_y)):max(0, c_y_pixel-int(0.1*d_y)), max(0, c_x_pixel-int(0.2*d_x)):min(explanation_size_x, c_x_pixel+int(0.2*d_x))] = 206
explanation_R[min(explanation_size_y, c_y_pixel+int(0.1*d_y)):min(explanation_size_y, c_y_pixel+int(0.2*d_y)), max(0, c_x_pixel-int(0.1*d_x)):min(explanation_size_x, c_x_pixel+int(0.2*d_x))] = 206

explanation_G[max(0, c_y_pixel+int(0.1*d_y)):min(explanation_size_y, c_y_pixel+int(0.2*d_y)), max(0, c_x_pixel-int(0.1*d_x)):min(explanation_size_x, c_x_pixel+int(0.2*d_x))] = 215
explanation_G[max(0, c_y_pixel-int(0.2*d_y)):min(explanation_size_y, c_y_pixel+int(0.2*d_y)), min(explanation_size_x, c_x_pixel+int(0.1*d_x)):min(explanation_size_x, c_x_pixel+int(0.2*d_x))] = 215
explanation_G[max(0, c_y_pixel-int(0.2*d_y)):max(0, c_y_pixel-int(0.1*d_y)), max(0, c_x_pixel-int(0.2*d_x)):min(explanation_size_x, c_x_pixel+int(0.2*d_x))] = 215
explanation_G[min(explanation_size_y, c_y_pixel+int(0.1*d_y)):min(explanation_size_y, c_y_pixel+int(0.2*d_y)), max(0, c_x_pixel-int(0.1*d_x)):min(explanation_size_x, c_x_pixel+int(0.2*d_x))] = 215

explanation_B[max(0, c_y_pixel+int(0.1*d_y)):min(explanation_size_y, c_y_pixel+int(0.2*d_y)), max(0, c_x_pixel-int(0.1*d_x)):min(explanation_size_x, c_x_pixel+int(0.2*d_x))] = 15
explanation_B[max(0, c_y_pixel-int(0.2*d_y)):min(explanation_size_y, c_y_pixel+int(0.2*d_y)), min(explanation_size_x, c_x_pixel+int(0.1*d_x)):min(explanation_size_x, c_x_pixel+int(0.2*d_x))] = 15
explanation_B[max(0, c_y_pixel-int(0.2*d_y)):max(0, c_y_pixel-int(0.1*d_y)), max(0, c_x_pixel-int(0.2*d_x)):min(explanation_size_x, c_x_pixel+int(0.2*d_x))] = 15
explanation_B[min(explanation_size_y, c_y_pixel+int(0.1*d_y)):min(explanation_size_y, c_y_pixel+int(0.2*d_y)), max(0, c_x_pixel-int(0.1*d_x)):min(explanation_size_x, c_x_pixel+int(0.2*d_x))] = 15


# THIRD
explanation_R[max(0, c_y_pixel-int(0.3*d_y)):min(explanation_size_y, c_y_pixel+int(0.3*d_y)), max(0, c_x_pixel-int(0.3*d_x)):max(0, c_x_pixel-int(0.2*d_x))] = 124
explanation_R[max(0, c_y_pixel-int(0.3*d_y)):min(explanation_size_y, c_y_pixel+int(0.3*d_y)), min(explanation_size_y, c_x_pixel+int(0.2*d_x)):min(explanation_size_y, c_x_pixel+int(0.3*d_x))] = 124
explanation_R[max(0, c_y_pixel-int(0.3*d_y)):max(0, c_y_pixel-int(0.2*d_y)), max(0, c_x_pixel-int(0.3*d_x)):min(explanation_size_y, c_x_pixel+int(0.3*d_x))] = 124
explanation_R[min(explanation_size_y, c_y_pixel+int(0.2*d_y)):min(explanation_size_y, c_y_pixel+int(0.3*d_y)), max(0, c_x_pixel-int(0.3*d_x)):min(explanation_size_y, c_x_pixel+int(0.3*d_x))] = 124

explanation_G[max(0, c_y_pixel-int(0.3*d_y)):min(explanation_size_y, c_y_pixel+int(0.3*d_y)), max(0, c_x_pixel-int(0.3*d_x)):max(0, c_x_pixel-int(0.2*d_x))] = 220
explanation_G[max(0, c_y_pixel-int(0.3*d_y)):min(explanation_size_y, c_y_pixel+int(0.3*d_y)), min(explanation_size_y, c_x_pixel+int(0.2*d_x)):min(explanation_size_y, c_x_pixel+int(0.3*d_x))] = 220
explanation_G[max(0, c_y_pixel-int(0.3*d_y)):max(0, c_y_pixel-int(0.2*d_y)), max(0, c_x_pixel-int(0.3*d_x)):min(explanation_size_y, c_x_pixel+int(0.3*d_x))] = 220
explanation_G[min(explanation_size_y, c_y_pixel+int(0.2*d_y)):min(explanation_size_y, c_y_pixel+int(0.3*d_y)), max(0, c_x_pixel-int(0.3*d_x)):min(explanation_size_y, c_x_pixel+int(0.3*d_x))] = 220

explanation_B[max(0, c_y_pixel-int(0.3*d_y)):min(explanation_size_y, c_y_pixel+int(0.3*d_y)), max(0, c_x_pixel-int(0.3*d_x)):max(0, c_x_pixel-int(0.2*d_x))] = 15
explanation_B[max(0, c_y_pixel-int(0.3*d_y)):min(explanation_size_y, c_y_pixel+int(0.3*d_y)), min(explanation_size_y, c_x_pixel+int(0.2*d_x)):min(explanation_size_y, c_x_pixel+int(0.3*d_x))] = 15
explanation_B[max(0, c_y_pixel-int(0.3*d_y)):max(0, c_y_pixel-int(0.2*d_y)), max(0, c_x_pixel-int(0.3*d_x)):min(explanation_size_y, c_x_pixel+int(0.3*d_x))] = 15
explanation_B[min(explanation_size_y, c_y_pixel+int(0.2*d_y)):min(explanation_size_y, c_y_pixel+int(0.3*d_y)), max(0, c_x_pixel-int(0.3*d_x)):min(explanation_size_y, c_x_pixel+int(0.3*d_x))] = 15


# FOURTH
explanation_R[max(0, c_y_pixel-int(0.4*d_y)):min(explanation_size_y, c_y_pixel+int(0.4*d_y)), max(0, c_x_pixel-int(0.4*d_x)):max(0, c_x_pixel-int(0.3*d_x))] = 108
explanation_R[max(0, c_y_pixel-int(0.4*d_y)):min(explanation_size_y, c_y_pixel+int(0.4*d_y)), min(explanation_size_y, c_x_pixel+int(0.3*d_x)):min(explanation_size_y, c_x_pixel+int(0.4*d_x))] = 108
explanation_R[max(0, c_y_pixel-int(0.4*d_y)):max(0, c_y_pixel-int(0.3*d_y)), max(0, c_x_pixel-int(0.4*d_x)):min(explanation_size_y, c_x_pixel+int(0.4*d_x))] = 108
explanation_R[min(explanation_size_y, c_y_pixel+int(0.3*d_y)):min(explanation_size_y, c_y_pixel+int(0.4*d_y)), max(0, c_x_pixel-int(0.4*d_x)):min(explanation_size_y, c_x_pixel+int(0.4*d_x))] = 108

explanation_G[max(0, c_y_pixel-int(0.4*d_y)):min(explanation_size_y, c_y_pixel+int(0.4*d_y)), max(0, c_x_pixel-int(0.4*d_x)):max(0, c_x_pixel-int(0.3*d_x))] = 196
explanation_G[max(0, c_y_pixel-int(0.4*d_y)):min(explanation_size_y, c_y_pixel+int(0.4*d_y)), min(explanation_size_y, c_x_pixel+int(0.3*d_x)):min(explanation_size_y, c_x_pixel+int(0.4*d_x))] = 196
explanation_G[max(0, c_y_pixel-int(0.4*d_y)):max(0, c_y_pixel-int(0.3*d_y)), max(0, c_x_pixel-int(0.4*d_x)):min(explanation_size_y, c_x_pixel+int(0.4*d_x))] = 196
explanation_G[min(explanation_size_y, c_y_pixel+int(0.3*d_y)):min(explanation_size_y, c_y_pixel+int(0.4*d_y)), max(0, c_x_pixel-int(0.4*d_x)):min(explanation_size_y, c_x_pixel+int(0.4*d_x))] = 196

explanation_B[max(0, c_y_pixel-int(0.4*d_y)):min(explanation_size_y, c_y_pixel+int(0.4*d_y)), max(0, c_x_pixel-int(0.4*d_x)):max(0, c_x_pixel-int(0.3*d_x))] = 8
explanation_B[max(0, c_y_pixel-int(0.4*d_y)):min(explanation_size_y, c_y_pixel+int(0.4*d_y)), min(explanation_size_y, c_x_pixel+int(0.3*d_x)):min(explanation_size_y, c_x_pixel+int(0.4*d_x))] = 8
explanation_B[max(0, c_y_pixel-int(0.4*d_y)):max(0, c_y_pixel-int(0.3*d_y)), max(0, c_x_pixel-int(0.4*d_x)):min(explanation_size_y, c_x_pixel+int(0.4*d_x))] = 8
explanation_B[min(explanation_size_y, c_y_pixel+int(0.3*d_y)):min(explanation_size_y, c_y_pixel+int(0.4*d_y)), max(0, c_x_pixel-int(0.4*d_x)):min(explanation_size_y, c_x_pixel+int(0.4*d_x))] = 8


# FIFTH
explanation_R[max(0, c_y_pixel-int(0.5*d_y)):min(explanation_size_y, c_y_pixel+int(0.5*d_y)), max(0, c_x_pixel-int(0.5*d_x)):max(0, c_x_pixel-int(0.4*d_x))] = 98
explanation_R[max(0, c_y_pixel-int(0.5*d_y)):min(explanation_size_y, c_y_pixel+int(0.5*d_y)), min(explanation_size_y, c_x_pixel+int(0.4*d_x)):min(explanation_size_y, c_x_pixel+int(0.5*d_x))] = 98
explanation_R[max(0, c_y_pixel-int(0.5*d_y)):max(0, c_y_pixel-int(0.4*d_y)), max(0, c_x_pixel-int(0.5*d_x)):min(explanation_size_y, c_x_pixel+int(0.5*d_x))] = 98
explanation_R[min(explanation_size_y, c_y_pixel+int(0.4*d_y)):min(explanation_size_y, c_y_pixel+int(0.5*d_y)), max(0, c_x_pixel-int(0.5*d_x)):min(explanation_size_y, c_x_pixel+int(0.5*d_x))] = 98

explanation_G[max(0, c_y_pixel-int(0.5*d_y)):min(explanation_size_y, c_y_pixel+int(0.5*d_y)), max(0, c_x_pixel-int(0.5*d_x)):max(0, c_x_pixel-int(0.4*d_x))] = 176
explanation_G[max(0, c_y_pixel-int(0.5*d_y)):min(explanation_size_y, c_y_pixel+int(0.5*d_y)), min(explanation_size_y, c_x_pixel+int(0.4*d_x)):min(explanation_size_y, c_x_pixel+int(0.5*d_x))] = 176
explanation_G[max(0, c_y_pixel-int(0.5*d_y)):max(0, c_y_pixel-int(0.4*d_y)), max(0, c_x_pixel-int(0.5*d_x)):min(explanation_size_y, c_x_pixel+int(0.5*d_x))] = 176
explanation_G[min(explanation_size_y, c_y_pixel+int(0.4*d_y)):min(explanation_size_y, c_y_pixel+int(0.5*d_y)), max(0, c_x_pixel-int(0.5*d_x)):min(explanation_size_y, c_x_pixel+int(0.5*d_x))] = 176

explanation_B[max(0, c_y_pixel-int(0.5*d_y)):min(explanation_size_y, c_y_pixel+int(0.5*d_y)), max(0, c_x_pixel-int(0.5*d_x)):max(0, c_x_pixel-int(0.4*d_x))] = 9
explanation_B[max(0, c_y_pixel-int(0.5*d_y)):min(explanation_size_y, c_y_pixel+int(0.5*d_y)), min(explanation_size_y, c_x_pixel+int(0.4*d_x)):min(explanation_size_y, c_x_pixel+int(0.5*d_x))] = 9
explanation_B[max(0, c_y_pixel-int(0.5*d_y)):max(0, c_y_pixel-int(0.4*d_y)), max(0, c_x_pixel-int(0.5*d_x)):min(explanation_size_y, c_x_pixel+int(0.5*d_x))] = 9
explanation_B[min(explanation_size_y, c_y_pixel+int(0.4*d_y)):min(explanation_size_y, c_y_pixel+int(0.5*d_y)), max(0, c_x_pixel-int(0.5*d_x)):min(explanation_size_y, c_x_pixel+int(0.5*d_x))] = 9


# RETURN FREE SPACE
explanation_R[temp == 120] = 120
explanation_G[temp == 120] = 120
explanation_B[temp == 120] = 120


explanation_R[int(chair1_cy-0.5*chair1_dy):int(chair1_cy+0.5*chair1_dy), int(chair1_cx-0.5*chair1_dx):int(chair1_cx+0.5*chair1_dx)] = 255
explanation_G[int(chair1_cy-0.5*chair1_dy):int(chair1_cy+0.5*chair1_dy), int(chair1_cx-0.5*chair1_dx):int(chair1_cx+0.5*chair1_dx)] = 0
explanation_B[int(chair1_cy-0.5*chair1_dy):int(chair1_cy+0.5*chair1_dy), int(chair1_cx-0.5*chair1_dx):int(chair1_cx+0.5*chair1_dx)] = 0


explanation = (np.dstack((explanation_R,explanation_G,explanation_B))).astype(np.uint8)



old_plan_xs = list(range(50,480,1)) 
old_plan_ys = [400]*len(old_plan_xs)

global_plan_xs = list(range(132,480,1))
global_plan_ys = [0.0064 * x**2 - 3.4113*x + 556.26 for x in global_plan_xs]
#print(global_plan_ys)
#pocetna = 250
#for i in range(0, 200):
#    global_plan_ys.append(pocetna - i)
#pocetna = global_plan_ys[-1]    
#for i in range(200, len(global_plan_xs)):
#    global_plan_ys.append(pocetna + i - 200)

fig = plt.figure(frameon=True)
w = 0.01 * explanation_size_x
h = 0.01 * explanation_size_y
fig.set_size_inches(w, h)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(explanation) #np.flip(explanation))#.astype(np.uint8))

C = np.array([217, 217, 217])
plt.plot(old_plan_xs, old_plan_ys, marker='.', c=C/255.0, markersize=6, alpha=0.1)

C = np.array([11, 240, 255])
plt.plot(global_plan_xs, global_plan_ys, marker='.', c=C/255.0, markersize=6, alpha=0.7)

ax.text(table_cx-30, table_cy+20, 'table', c='white', fontsize=17, fontweight=20.0)
ax.text(chair1_cx-35, chair1_cy-20, 'chair_1', c='white', fontsize=15, fontweight=20.0)
ax.text(chair2_cx-35, chair2_cy+20, 'chair_2', c='white', fontsize=15, fontweight=20.0)
ax.text(bookshelf_cx-40, bookshelf_cy+10, 'bookshelf', c='white', fontsize=17, fontweight=20.0)
ax.text(85, 275, 'robot', c='white', fontsize=15, fontweight=20.0)
ax.quiver([115], [250], [0.5], [0.8], color='white', alpha=1.0)
ax.text(300, 580, 'wall', c='white', fontsize=17, fontweight=20.0)


# CONVERT IMAGE TO NUMPY ARRAY 
fig.savefig('explanation' + '.png', transparent=False)
plt.close()

#from PIL import Image
#im = Image.fromarray(explanation)
#im.save("explanation.jpg")