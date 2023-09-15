#!/usr/bin/env python3

import pandas as pd
from scipy.stats import ttest_ind
import numpy as np

i = pd.read_csv('i.csv')
e = pd.read_csv('e.csv')


#age = pd.concat([i["Age"], e["Age"]], axis = 0)
#print('age statistics:', age.describe())

#sex = pd.concat([i["Sex"], e["Sex"]], axis = 0)
#print('sex statistics:', sex.describe())
#print(np.unique(sex))


# get all values in one column
i_all = pd.concat([i.iloc[:, 18], i.iloc[:, 19], i.iloc[:, 20], i.iloc[:, 21], i.iloc[:, 22], i.iloc[:, 23], i.iloc[:, 24], i.iloc[:, 25]], axis = 0)
print('i_median: ', i_all.median())
print('i_statistics: ', i_all.describe())

e_all = pd.concat([e.iloc[:, 18], e.iloc[:, 19], e.iloc[:, 20], e.iloc[:, 21], e.iloc[:, 22], e.iloc[:, 23], e.iloc[:, 24], e.iloc[:, 25]], axis = 0)
print('e_median: ', e_all.median())
print('e_statistics: ', e_all.describe())

test = ttest_ind(i_all, e_all)
print('t_test all = ', test)

'''
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
'''
#test = ttest_ind(i[i.columns[18:27]], e[e.columns[18:27]])
#print(str(test))
