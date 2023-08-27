#!/usr/bin/env python3

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image

#visual_time,visual_N,textual_time,textual_N,N_objects,N_words,N_deviations

features = ['extroversion','visual_time','visual_N','textual_time','textual_N','N_objects','N_words','N_deviations']

averages = []
#averages.append(features)

for i in range(0, 11):
    df = pd.read_csv(str(i) + '.csv')

    extroversion = i * 10
    visual_time_avg = df['visual_time'].mean()
    visual_N_avg = df["visual_N"].iloc[-1]
    textual_time_avg = df['textual_time'].mean()
    textual_N_avg = df["textual_N"].iloc[-1]
    N_objects_avg = df["N_objects"].sum() / visual_N_avg
    N_words_avg = df["N_words"].sum() / textual_N_avg
    N_deviations_avg = df["N_deviations"].iloc[-1]

    averages.append([extroversion,visual_time_avg,visual_N_avg,textual_time_avg,textual_N_avg,N_objects_avg,N_words_avg,N_deviations_avg])

#print(averages)
df = pd.DataFrame(averages)

df.columns = features 
#df.set_index('extroversion', inplace=True)

print(df)

fig = plt.figure(frameon=True)
#w = 0.01 * explanation_size_x
#h = 0.01 * explanation_size_y
#fig.set_size_inches(w, h)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

plt.plot(df['extroversion'].tolist(), df['visual_N'].tolist(), color='red', marker='.', markersize=6, alpha=0.7)
plt.plot(df['extroversion'].tolist(), df['textual_N'].tolist(), color='blue', marker='.', markersize=6, alpha=0.7)

plt.show()