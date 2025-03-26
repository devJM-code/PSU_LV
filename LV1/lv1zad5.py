rjecnik = {}

try:
with open("song.txt", 'r') as datoteka:
for linija in datoteka:
rijeci = linija.split()
for rijec in rijeci:
rijec = rijec.lower()
if rijec in rjecnik:
rjecnik[rijec] += 1
else:
rjecnik[rijec] = 1

except FileNotFoundError:
print("Datoteka nije pronađena.")

unikatne_rijeci = [rijec for rijec, broj in rjecnik.items() if broj == 1]

print("Broj riječi koje se pojavljuju samo jednom:", len(unikatne_rijeci))
print("Riječi koje se pojavljuju samo jednom:", unikatne_rijeci)
