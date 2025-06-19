import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.cluster.hierarchy import dendrogram, linkage

def generiraj_skup_podataka(broj_primjera, tip_skupa):
   
    if tip_skupa == 1:
        
        podaci, _ = datasets.make_blobs(
            n_samples=broj_primjera, 
            random_state=365
        )
    elif tip_skupa == 2:
       
        podaci, _ = datasets.make_blobs(
            n_samples=broj_primjera, 
            random_state=148
        )
        transformacija = [
            [0.60834549, -0.63667341],
            [-0.40887718, 0.85253229]
        ]
        podaci = np.dot(podaci, transformacija)
    elif tip_skupa == 3:
        
        podaci, _ = datasets.make_blobs(
            n_samples=broj_primjera,
            centers=4,
            cluster_std=[1.0, 2.5, 0.5, 3.0],
            random_state=148
        )
    elif tip_skupa == 4:
        
        podaci, _ = datasets.make_circles(
            n_samples=broj_primjera,
            factor=0.5,
            noise=0.05
        )
    elif tip_skupa == 5:
        
        podaci, _ = datasets.make_moons(
            n_samples=broj_primjera,
            noise=0.05
        )
    else:
        podaci = np.array([])
    
    return podaci

def pripremi_hijerarhijsko_klasteriranje(podaci, metoda="ward"):
   
    return linkage(podaci, method=metoda)

def vizualiziraj_dendrogram(matrica_povezivanja, broj_primjera):

    plt.figure(figsize=(12, 8))

    labele = [f"Primjer {i+1}" for i in range(broj_primjera)]
    
    dendrogram(
        matrica_povezivanja,
        labels=labele,
        orientation="top",
        leaf_rotation=90,
        leaf_font_size=10,
        show_contracted=True
    )
    
    plt.title("Hijerarhijsko klasteriranje - Dendrogram", fontsize=14)
    plt.xlabel("Indeks podatkovnog primjera", fontsize=12)
    plt.ylabel("Udaljenost između klastera", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
   
    BROJ_PRIMJERA = 25
    TIP_PODATAKA = 5  
    METODA_POVEZIVANJA = "ward"
 
    podaci = generiraj_skup_podataka(BROJ_PRIMJERA, TIP_PODATAKA)

    matrica_povezivanja = pripremi_hijerarhijsko_klasteriranje(
        podaci, 
        METODA_POVEZIVANJA
    )

    vizualiziraj_dendrogram(matrica_povezivanja, BROJ_PRIMJERA)

if __name__ == "__main__":
    main()
