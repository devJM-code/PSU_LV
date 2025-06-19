import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

def generiraj_skup_podataka(broj_primjera, tip_skupa):
 
    if tip_skupa == 1:
     
        podaci, _ = datasets.make_blobs(n_samples=broj_primjera, random_state=365)
    elif tip_skupa == 2:
        
        podaci, _ = datasets.make_blobs(n_samples=broj_primjera, random_state=148)
        transformacija = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
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

def analiziraj_optimalni_k(podaci, max_k=20):
  
    kriterijske_vrijednosti = []
    brojevi_klastera = range(2, max_k + 1)
    
    for k in brojevi_klastera:
        model = KMeans(n_clusters=k, random_state=0)
        model.fit(podaci)
        kriterijske_vrijednosti.append(model.inertia_)
    
    return kriterijske_vrijednosti

def vizualiziraj_metodu_lakta(brojevi_klastera, kriterijske_vrijednosti):
  
    plt.figure(figsize=(10, 6))
    plt.plot(
        brojevi_klastera,
        kriterijske_vrijednosti,
        marker='o',
        markersize=8,
        markerfacecolor='white',
        markeredgewidth=2
    )
    
    plt.xlabel('Broj klastera (K)', fontsize=12)
    plt.ylabel('Kriterijska vrijednost (inertia)', fontsize=12)
    plt.title('Metoda lakta za određivanje optimalnog K', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(brojevi_klastera)
    plt.tight_layout()
    plt.show()

def main():
   
    BROJ_PRIMJERA = 500
    TIP_PODATAKA = 5  
    MAKSIMALNI_K = 20

    podaci = generiraj_skup_podataka(BROJ_PRIMJERA, TIP_PODATAKA)

    kriterijske_vrijednosti = analiziraj_optimalni_k(podaci, MAKSIMALNI_K)
    
    vizualiziraj_metodu_lakta(range(2, MAKSIMALNI_K + 1), kriterijske_vrijednosti)

if __name__ == "__main__":
    main()
