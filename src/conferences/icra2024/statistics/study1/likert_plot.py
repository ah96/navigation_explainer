#!pip install plot-likert

import pandas as pd
import numpy as np
import plot_likert
import matplotlib.pyplot as plt

t = pd.read_csv('t_str_q.csv',header=0)
v = pd.read_csv('v_str_q.csv',header=0)
vt = pd.read_csv('vt_str_q.csv',header=0)

t_likert = t.iloc[:,17:25]
#print(t_likert)

vt_likert = vt.iloc[:,17:25]

v_likert = v.iloc[:,17:25]

my_scale = \
    ['Strongly disagree',
     'Somewhat disagree',
     'Neutral',
     'Somewhat agree',
     'Strongly agree']

questions_asked = \
    ['I am able to understand the actions/behavior of the robot with the given explanation.',
     'I am satisfied with the explanation provided.',
     'The explanation provides sufficient details of the robot\'s actions and behaviors.',
     'The explanation accurately describes the movement and actions of the robot.',
     'The explanation provides reliable information about the robot\'s actions.',
	 'I could see the reasons behind choosing this method for the explanation.'
	 'The explanation describes the robot\'s actions/behaviors efficiently.',
	 'The explanation describes the robot\'s actions and situation completely.'
	]

#plot_likert.plot_likert(t_likert, plot_likert.scales.agree, plot_percentage=True)

# textual
#axes = plot_likert.plot_likert(t_likert, my_scale, plot_percentage=True,figsize=(25,10))
#axes.get_figure().savefig('textual_likert.png')

# hybrid
#axes = plot_likert.plot_likert(vt_likert, my_scale, plot_percentage=True)
#axes.get_figure().savefig('visual_textual_likert.png')

# visual
#axes = plot_likert.plot_likert(v_likert, my_scale, plot_percentage=True)
#axes.get_figure().savefig('visual_likert.png')


# textual_vs_hybrid
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,5))

ax_1 = plot_likert.plot_likert(df=t_likert, plot_scale=my_scale, plot_percentage=True, xtick_interval=10, ax=ax1, legend=0)
ax1.set_title("Group 1")#, weight='bold',linewidth=2,fontsize=20)

ax_2 = plot_likert.plot_likert(df=vt_likert, plot_scale=my_scale, plot_percentage=True, xtick_interval=10, ax=ax2, legend=0)
ax2.set_title("Group 2")

# display a single legend for the whole figure
handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(1.08, .9))
#plt.show()
plt.savefig('textual_vs_hybrid_likert.svg', bbox_inches='tight', format="svg")