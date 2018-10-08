import pandas as pd

df = pd.read_csv('Galton.txt', delimiter="\t")

in2cm = 2.54

cols_to_update = ["Height", "Father", "Mother"]

for c in cols_to_update:
    df[c] = df[c].apply(lambda x: round(x * in2cm))

print(df)


