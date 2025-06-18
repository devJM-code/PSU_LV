import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/mnt/data/mtcars.csv', index_col=0)

sns.set(style="whitegrid")

plt.figure(figsize=(8, 6))
avg_mpg = df.groupby('cyl')['mpg'].mean().reset_index()
sns.barplot(data=avg_mpg, x='cyl', y='mpg', palette='viridis')
plt.title('Prosječna potrošnja po broju cilindara')
plt.xlabel('Broj cilindara')
plt.ylabel('Potrošnja (mpg)')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='cyl', y='wt', palette='pastel')
plt.title('Distribucija težine po broju cilindara')
plt.xlabel('Broj cilindara')
plt.ylabel('Težina (1000 lbs)')
plt.show()

plt.figure(figsize=(8, 6))
df['am'] = df['am'].map({0: 'Automatski', 1: 'Ručni'})
sns.boxplot(data=df, x='am', y='mpg', palette='Set2')
plt.title('Potrošnja prema vrsti mjenjača')
plt.xlabel('Vrsta mjenjača')
plt.ylabel('Potrošnja (mpg)')
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='hp', y='qsec', hue='am', style='am', palette='Dark2', s=100)
plt.title('Ubrzanje vs Snaga za različite mjenjače')
plt.xlabel('Snaga (hp)')
plt.ylabel('Ubrzanje (qsec)')
plt.legend(title='Mjenjač')
plt.show()
