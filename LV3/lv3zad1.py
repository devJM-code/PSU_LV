import pandas as pd

df = pd.read_csv('/mnt/data/mtcars.csv', index_col=0)

najveca_potrosnja = df.sort_values('mpg').head(5)
print("1. 5 automobila s najvećom potrošnjom:")
print(najveca_potrosnja[['mpg']])

najmanja_potrosnja_8cil = df[df['cyl'] == 8].sort_values('mpg').head(3)
print("\n2. Tri automobila s 8 cilindara i najmanjom potrošnjom:")
print(najmanja_potrosnja_8cil[['mpg']])

srednja_6cil = df[df['cyl'] == 6]['mpg'].mean()
print("\n3. Srednja potrošnja automobila sa 6 cilindara:", round(srednja_6cil, 2))

srednja_4cil_masa = df[(df['cyl'] == 4) & (df['wt'] * 1000 >= 2000) & (df['wt'] * 1000 <= 2200)]['mpg'].mean()
print("\n4. Srednja potrošnja 4-cilindraša mase 2000–2200 lbs:", round(srednja_4cil_masa, 2))

broj_mjenjaca = df['am'].value_counts()
print("\n5. Broj automobila po vrsti mjenjača:")
print("Automatski (0):", broj_mjenjaca.get(0, 0))
print("Ručni (1):", broj_mjenjaca.get(1, 0))

auto_preko_100hp = df[(df['am'] == 0) & (df['hp'] > 100)].shape[0]
print("\n6. Broj automobila s automatskim mjenjačem i više od 100 KS:", auto_preko_100hp)

df['masa_kg'] = df['wt'] * 1000 * 0.453592
print("\n7. Masa svakog automobila u kilogramima:")
print(df['masa_kg'])
