try:
ocijena = float(input("Unesite float ocijenu od 0.0 do 1.0 \n"))
if not (0.0 <= ocijena <= 1.0):
print("Error broj je izvan podrucja")
ocijena = float(ocijena)
except:
print("Unesite samo brojeve float")

if(ocijena >= 0.9):
print('A')
elif(ocijena >= 0.8):
print('B')
elif(ocijena >= 0.7):
print('C')
elif(ocijena >= 0.6):
print('D')
elif(ocijena < 0.6):
print('F')
