#!/usr/bin/env python3

import pandas as pd
from scipy.stats import ttest_ind

#create pandas DataFrame
#df = pd.DataFrame({'method': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A',
#                              'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'],
#                   'score': [71, 72, 72, 75, 78, 81, 82, 83, 89, 91, 80, 81, 81,
#                             84, 88, 88, 89, 90, 90, 91]})

t = pd.read_csv('t.csv')
v = pd.read_csv('v.csv')
vt = pd.read_csv('vt.csv')

#view first five rows of DataFrame
#t.head()
#print(t)

#define samples
#group1 = df[df['method']=='A']
#group2 = df[df['method']=='B']

#print(v[v.columns[17]])

#perform independent two sample t-test
print('\nt-test: textual versus visual:')
test = ttest_ind(t[t.columns[17]], v[v.columns[17]])
print("'" + t.columns[17] + "'" + "  " + str(test))
test = ttest_ind(t[t.columns[18]], v[v.columns[18]])
print("'" + t.columns[18] + "'" + "  " + str(test))
test = ttest_ind(t[t.columns[19]], v[v.columns[19]])
print("'" + t.columns[19] + "'" + "  " + str(test))
test = ttest_ind(t[t.columns[20]], v[v.columns[20]])
print("'" + t.columns[20] + "'" + "  " + str(test))
test = ttest_ind(t[t.columns[21]], v[v.columns[21]])
print("'" + t.columns[21] + "'" + "  " + str(test))
test = ttest_ind(t[t.columns[22]], v[v.columns[22]])
print("'" + t.columns[22] + "'" + "  " + str(test))
test = ttest_ind(t[t.columns[23]], v[v.columns[23]])
print("'" + t.columns[23] + "'" + "  " + str(test))
test = ttest_ind(t[t.columns[24]], v[v.columns[24]])
print("'" + t.columns[24] + "'" + "  " + str(test))
test = ttest_ind(t[t.columns[25]], v[v.columns[25]])
print("'" + t.columns[25] + "'" + "  " + str(test))

print('\nt-test: textual versus visual-textual:')
test = ttest_ind(t[t.columns[17]], vt[vt.columns[17]])
print("'" + t.columns[17] + "'" + "  " + str(test))
test = ttest_ind(t[t.columns[18]], vt[vt.columns[18]])
print("'" + t.columns[18] + "'" + "  " + str(test))
test = ttest_ind(t[t.columns[19]], vt[vt.columns[19]])
print("'" + t.columns[19] + "'" + "  " + str(test))
test = ttest_ind(t[t.columns[20]], vt[vt.columns[20]])
print("'" + t.columns[20] + "'" + "  " + str(test))
test = ttest_ind(t[t.columns[21]], vt[vt.columns[21]])
print("'" + t.columns[21] + "'" + "  " + str(test))
test = ttest_ind(t[t.columns[22]], vt[vt.columns[22]])
print("'" + t.columns[22] + "'" + "  " + str(test))
test = ttest_ind(t[t.columns[23]], vt[vt.columns[23]])
print("'" + t.columns[23] + "'" + "  " + str(test))
test = ttest_ind(t[t.columns[24]], vt[vt.columns[24]])
print("'" + t.columns[24] + "'" + "  " + str(test))
test = ttest_ind(t[t.columns[25]], vt[vt.columns[25]])
print("'" + t.columns[25] + "'" + "  " + str(test))

print('\nt-test: visual versus visual-textual:')
test = ttest_ind(v[v.columns[17]], vt[vt.columns[17]])
print("'" + t.columns[17] + "'" + "  " + str(test))
test = ttest_ind(v[v.columns[18]], vt[vt.columns[18]])
print("'" + t.columns[18] + "'" + "  " + str(test))
test = ttest_ind(v[v.columns[19]], vt[vt.columns[19]])
print("'" + t.columns[19] + "'" + "  " + str(test))
test = ttest_ind(v[v.columns[20]], vt[vt.columns[20]])
print("'" + t.columns[20] + "'" + "  " + str(test))
test = ttest_ind(v[v.columns[21]], vt[vt.columns[21]])
print("'" + t.columns[21] + "'" + "  " + str(test))
test = ttest_ind(v[v.columns[22]], vt[vt.columns[22]])
print("'" + t.columns[22] + "'" + "  " + str(test))
test = ttest_ind(v[v.columns[23]], vt[vt.columns[23]])
print("'" + t.columns[23] + "'" + "  " + str(test))
test = ttest_ind(v[v.columns[24]], vt[vt.columns[24]])
print("'" + t.columns[24] + "'" + "  " + str(test))
test = ttest_ind(v[v.columns[25]], vt[vt.columns[25]])
print("'" + t.columns[25] + "'" + "  " + str(test))

#Ttest_indResult(statistic=-2.6034304605397938, pvalue=0.017969284594810425)