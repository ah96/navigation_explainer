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
    visual_time_avg = df['visual_time'].mean() * 1000
    visual_N_avg = df["visual_N"].iloc[-1]
    textual_time_avg = df['textual_time'].mean() * 1000
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

# plot number (N) of visual and textual explanations
fig = plt.figure(frameon=True)
#plt.margins(x=0, y=0)
#w = 0.01 * explanation_size_x
#h = 0.01 * explanation_size_y
#fig.set_size_inches(w, h)
#ax = plt.Axes(fig, [0., 0., 1., 1.])
#ax.set_axis_off()
#fig.add_axes(ax)

plt.plot(df['extroversion'].tolist(), df['visual_N'].tolist(), color='red', marker='o', markersize=12, alpha=1.)
plt.plot(df['extroversion'].tolist(), df['textual_N'].tolist(), color='blue', marker='.', markersize=12, alpha=1.)

plt.legend(['visual explanations', 'textual explanations'])

#plt.xticks(np.arange(0, 11, step=1))
#plt.xticks(np.arange(0, 11, step=1))

#ax.set_xlabel
plt.xlabel('extroversion level (percentage)')
plt.ylabel('number of explanations generated')

fig.tight_layout()

plt.savefig('eval_N_exps.png')
plt.savefig('eval_N_exps.eps', format='eps')

#plt.show()


# plot visual and textual average runtimes
fig = plt.figure(frameon=True)
#plt.margins(x=0, y=0)
#w = 0.01 * explanation_size_x
#h = 0.01 * explanation_size_y
#fig.set_size_inches(w, h)
#ax = plt.Axes(fig, [0., 0., 1., 1.])
#ax.set_axis_off()
#fig.add_axes(ax)

plt.plot(df['extroversion'].tolist(), df['visual_time'].tolist(), color='red', marker='o', markersize=12, alpha=1.)
plt.plot(df['extroversion'].tolist(), df['textual_time'].tolist(), color='blue', marker='.', markersize=12, alpha=1.)

plt.legend(['visual explanations', 'textual explanations'])

#plt.xticks(np.arange(0, 11, step=1))
#plt.xticks(np.arange(0, 11, step=1))

#ax.set_xlabel
plt.xlabel('extroversion level (percentage)')
plt.ylabel('average explanation runtime (milliseconds)')

fig.tight_layout()

plt.savefig('eval_runtimes.png')
plt.savefig('eval_runtimes.eps', format='eps')

#plt.show()



# plot number (N) of visual and textual explanations
fig = plt.figure(frameon=True)
#plt.margins(x=0, y=0)
#w = 0.01 * explanation_size_x
#h = 0.01 * explanation_size_y
#fig.set_size_inches(w, h)
#ax = plt.Axes(fig, [0., 0., 1., 1.])
#ax.set_axis_off()
#fig.add_axes(ax)

plt.plot(df['extroversion'].tolist(), df['N_objects'].tolist(), color='red', marker='o', markersize=12, alpha=1.)
plt.plot(df['extroversion'].tolist(), df['N_words'].tolist(), color='blue', marker='.', markersize=12, alpha=1.)
plt.plot(df['extroversion'].tolist(), df['N_deviations'].tolist(), color='green', marker='x', markersize=12, alpha=1.)

plt.legend(['average number of objects in visual explanations', 'average number of words in textual explanations', 'number of deviations explained'])

#plt.xticks(np.arange(0, 11, step=1))
#plt.xticks(np.arange(0, 11, step=1))

#ax.set_xlabel
plt.xlabel('extroversion level (percentage)')
#plt.ylabel('number of explanations generated')

fig.tight_layout()

plt.savefig('eval_N_artefacts.png')
plt.savefig('eval_N_artefacts.eps', format='eps')

#plt.show()