import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def ucitaj_sliku(putanja):
 
    slika = plt.imread(putanja)
    if len(slika.shape) == 3: 
        slika = np.mean(slika, axis=2)  
    return slika

def kvantiziraj_sliku(slika, broj_klastera=10):

    X = slika.reshape((-1, 1))
    
  
    model = KMeans(n_clusters=broj_klastera, n_init=10)
    model.fit(X)
    
   
    centri = model.cluster_centers_.squeeze()
    labele = model.labels_
    slika_kvantizirana = np.choose(labele, centri).reshape(slika.shape)
    
    return slika_kvantizirana, centri

def prikazi_slike(originalna, kvantizirana, broj_klastera):

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(originalna, cmap='gray', vmin=0, vmax=255)
    plt.title('Originalna slika')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(kwantizirana, cmap='gray', vmin=0, vmax=255)
    plt.title(f'Kvantizirana slika ({broj_klastera} nijansi)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def izracunaj_kompresiju(broj_klastera):
 
    originalni_bits = 8
    komprimirani_bits = int(np.ceil(np.log2(broj_klastera)))
    omjer_kompresije = (komprimirani_bits / originalni_bits) * 100
    return omjer_kompresije

def main():

    PUTANJA_SLIKE = 'example_grayscale.png'
    BROJ_NIJANSI = 10 

    try:
        img_original = ucitaj_sliku(PUTANJA_SLIKE)
    except FileNotFoundError:
        print(f"Greška: Datoteka '{PUTANJA_SLIKE}' nije pronađena.")
        return

    img_kvantizirana, _ = kvantiziraj_sliku(img_original, BROJ_NIJANSI)

    prikazi_slike(img_original, img_kvantizirana, BROJ_NIJANSI)
 
    kompresija = izracunaj_kompresiju(BROJ_NIJANSI)
    print(f"Teoretski omjer kompresije: {kompresija:.2f}% originalne veličine")
    print(f"Broj korištenih nijansi: {BROJ_NIJANSI}")

if __name__ == "__main__":
    main()
