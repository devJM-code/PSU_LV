import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.io import imread, imsave

def ucitaj_sliku(putanja):

    slika = imread(putanja)
    if len(slika.shape) != 3 or slika.shape[2] != 3:
        raise ValueError("Slika mora biti u RGB formatu")
    return slika

def kvantiziraj_boje(slika, broj_boja=12):

    visina, sirina = slika.shape[:2]
    pikseli = slika.reshape(-1, 3)

    model = KMeans(n_clusters=broj_boja, random_state=42)
    model.fit(pikseli)
    
    kvantizirani_pikseli = model.cluster_centers_[model.labels_]
    return kvantizirani_pikseli.reshape((visina, sirina, 3)).astype(np.uint8)

def prikazi_rezultate(originalna, kvantizirana):

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(originalna)
    plt.title('Originalna slika', fontsize=12)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(kwantizirana)
    plt.title(f'Kvantizirana slika ({kvantizirana.max()} boja)', fontsize=12)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():

    PUTANJA_ULAZ = "example.png"
    PUTANJA_IZLAZ = "quantized_example.png"
    BROJ_BOJA = 12
    
    try:

        original = ucitaj_sliku(PUTANJA_ULAZ)
        kvantizirana = kvantiziraj_boje(original, BROJ_BOJA)

        imsave(PUTANJA_IZLAZ, kvantizirana)
        prikazi_rezultate(original, kvantizirana)
        
        print(f"Slika uspješno kvantizirana i spremljena kao '{PUTANJA_IZLAZ}'")
        print(f"Broj korištenih boja: {BROJ_BOJA}")
        
    except Exception as e:
        print(f"Došlo je do greške: {str(e)}")

if __name__ == "__main__":
    main()
