spam_counter = 0
sum = 0.0

unos = input("Unesite ime tekstualne datoteke \n")
datoteka = 'C:\\Users\\student\\Desktop\\' + unos + '.txt'

try:
dat = open(datoteka, 'r')
except:
print("Datoteka ne postoji")

for line in dat:
if "X-DSPAM-Confidence:" in line:
spam_counter += 1
sum += float(line.split(":")[1])

dat.close()

if spam_counter > 0:
print(f"Srednja vrijednost pouzdanosti je: {sum / spam_counter}")
else:
print("Nema pronađenih linija s X-DSPAM-Confidence.")
