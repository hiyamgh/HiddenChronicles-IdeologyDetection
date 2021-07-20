import pandas as pd

df = pd.read_csv('DECO_v.1.0.csv')
print(len(df))
df = df[df['conflict_name'].str.contains("Israel") | df['conflict_name'].str.contains("Fatah")]
df.to_csv('DECO_v.1.0_filtered.csv', index=False)
print(len(df))