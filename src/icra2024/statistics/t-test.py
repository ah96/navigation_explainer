#!/usr/bin/env python3

import pandas as pd
from scipy.stats import ttest_ind
import numpy as np

t = pd.read_csv('t.csv')
v = pd.read_csv('v.csv')
vt = pd.read_csv('vt.csv')


#age = pd.concat([v["Age"], vt["Age"]], axis = 0)
#print('age statistics:', age.describe())

#sex = pd.concat([v["Sex"], vt["Sex"]], axis = 0)
#print('sex statistics:', sex.describe())
#print(np.unique(sex))

# get all values in one column
v_all = pd.concat([v.iloc[:, 17], v.iloc[:, 18], v.iloc[:, 19], v.iloc[:, 20], v.iloc[:, 21], v.iloc[:, 22], v.iloc[:, 23], v.iloc[:, 24]], axis = 0)
print('v_median: ', v_all.median())
print('v_statistics: ', v_all.describe())

vt_all = pd.concat([vt.iloc[:, 17], vt.iloc[:, 18], vt.iloc[:, 19], vt.iloc[:, 20], vt.iloc[:, 21], vt.iloc[:, 22], vt.iloc[:, 23], vt.iloc[:, 24]], axis = 0)
print('vt_median: ', vt_all.median())
print('vt_statistics: ', vt_all.describe())

test = ttest_ind(v_all, vt_all)
print('t_test all = ', test)

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
test = ttest_ind(t[t.columns[17:25]], v[v.columns[17:25]])
print(str(test))

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
test = ttest_ind(v[v.columns[17:25]], vt[vt.columns[17:25]])
print(str(test))

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
test = ttest_ind(t[t.columns[17:25]], vt[vt.columns[17:25]])
print(str(test))

