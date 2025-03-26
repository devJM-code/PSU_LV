lst = []
counter = 0
while True:
try:
unos = input("unesi broj \n")
if( unos == "Done"):
break
else:
broj = int(unos)
lst.append(broj)
counter += 1

except:
print("Unositi samo brojeve !!")

print("Korisnik je unjeo :", counter, "brojeva")
avg = float(sum(lst)/len(lst))
print("Srednja vrijednost: ",avg)
print('minimalna vrijednost: ', min(lst))
print('maksimalna vrijednost: ', max(lst))
lst.sort()
print(lst)
