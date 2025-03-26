def total_euro(Radni_sati, Zarada):
return Radni_sati * Zarada

Radni_sati = int(input("Unos radnih sati: \n"))
Zarada = float(input("Unos zarade: \n"))

print('Radni_sati: ', Radni_sati)
print('Zarada: ', Zarada)

Ukupno = total_euro(Radni_sati, Zarada)
print('Ukupno: ', Ukupno)
