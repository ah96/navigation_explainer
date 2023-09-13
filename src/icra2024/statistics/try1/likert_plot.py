#!/usr/bin/env python3

# Make sure you have some data
import pandas as pd

t = pd.read_csv('t_str.csv',header=0)
v = pd.read_csv('v_str.csv',header=0)
vt = pd.read_csv('vt_str.csv',header=0)

t_likert = t.iloc[:,17:26]
#print(t_likert)

vt_likert = vt.iloc[:,17:26]

v_likert = v.iloc[:,17:26]

# Now plot it!
import plot_likert
import matplotlib.pyplot as plt

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
	 'I could see the reasons behind choosing this method (textual) for the explanation.'
	 'The explanation describes the robot\'s actions/behaviors efficiently.',
     'With the explanation provided, I am able to predict the behavior of the robot.',
	 'The explanation describes the robot\'s actions and situation completely.'
	]

#plot_likert.plot_likert(t_likert, plot_likert.scales.agree, plot_percentage=True)

# textual
axes = plot_likert.plot_likert(t_likert, my_scale, plot_percentage=True,figsize=(25,10))
axes.get_figure().savefig('textual_likert.png')

# hybrid
axes = plot_likert.plot_likert(vt_likert, my_scale, plot_percentage=True)
axes.get_figure().savefig('visual_textual_likert.png')

# visual
axes = plot_likert.plot_likert(v_likert, my_scale, plot_percentage=True)
axes.get_figure().savefig('visual_likert.png')


# textual_vs_hybrid
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,3))
ax_1 = plot_likert.plot_likert(t_likert, my_scale, 
                        plot_percentage=True,  # show absolute values
                        legend=0,  # hide the legend for the subplot, we'll show a single figure legend instead                        
                       )
ax_2 = plot_likert.plot_likert(vt_likert, my_scale, 
                        plot_percentage=True,  # show percentage values
                        legend=0,  # hide the legend for the subplot, we'll show a single figure legend instead
                        #width=0.15  # make the bars slimmer
                       )

# display a single legend for the whole figure
legend, handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(1.08, .9))
#plt.show()
plt.savefig('textual_vs_hybrid_likert.png')