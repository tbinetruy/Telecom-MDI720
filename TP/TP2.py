import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.kde import KDEUnivariate

###
### Q1
###

#
# 1
#

df = pd.read_csv('Galton.txt', delimiter="\t")

in2cm = 2.54

cols_to_update = ["Height", "Father", "Mother"]

for c in cols_to_update:
    df[c] = df[c].apply(lambda x: x * in2cm)

#
# 2
#

print("There are " + str(df.isnull().sum().sum()) + " null data in the tale")

#
# 3
#

plt.figure()
kde_father = KDEUnivariate(df['Father'])
kde_father.fit(bw=2, kernel='gau')
x_grid = np.linspace(140, 210)
pdf_est_father = kde_father.evaluate(x_grid)

kde_mother = KDEUnivariate(df['Mother'])
kde_mother.fit(bw=2, kernel='gau')
x_grid = np.linspace(140, 210)
pdf_est_mother = kde_mother.evaluate(x_grid)

my_blue = 'blue'
my_orange = 'orange'

plt.plot(x_grid, pdf_est_father, color=my_blue, label='Father')
plt.fill_between(x_grid, 0, pdf_est_father, facecolor=my_blue, alpha=0.5)

plt.plot(x_grid, pdf_est_mother, color=my_orange, label='Mother')
plt.fill_between(x_grid, 0, pdf_est_mother, facecolor=my_orange, alpha=0.5)

plt.ylabel('Density', fontsize=18)
plt.xlabel('Height (in cm.)', fontsize=18)
plt.title("Distribution of parents height (by gender)")
plt.legend()
plt.show()
