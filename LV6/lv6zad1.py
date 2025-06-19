import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

NUM_SAMPLES = 500
NUM_CLUSTERS = 3
RANDOM_STATE = 42
DATA_TYPE = 5  

def generiraj_podatke(broj_primjera, tip_podataka):

    if tip_podataka == 1:

        podaci, _ = datasets.make_blobs(n_samples=broj_primjera, random_state=365)
    elif tip_podataka == 2:

        podaci, _ = datasets.make_blobs(n_samples=broj_primjera, random_state=148)
        transformacija = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        podaci = np.dot(podaci, transformacija)
    elif tip_podataka == 3:

        podaci, _ = datasets.make_blobs(n_samples=broj_primjera, centers=4,
                                      cluster_std=[1.0, 2.5, 0.5, 3.0], random_state=148)
    elif tip_podataka == 4:

        podaci, _ = datasets.make_circles(n_samples=broj_primjera, factor=0.5, noise=0.05)
    elif tip_podataka == 5:

        podaci, _ = datasets.make_moons(n_samples=broj_primjera, noise=0.05)
    else:
        podaci = np.empty((0, 2))
        
    return podaci

def vizualiziraj_rezultate(podaci, labele, centri, tip_podataka):
  
    plt.figure(figsize=(10, 7))
    
    plt.scatter(podaci[:, 0], podaci[:, 1], c=labele, cmap='plasma', 
               s=50, alpha=0.6, edgecolor='k')
    
    plt.scatter(centri[:, 0], centri[:, 1], c='white', marker='X',
               s=200, linewidths=2, edgecolor='black', label='Centri klastera')
    
    plt.title(f'K-means klasteriranje (Tip podataka: {tip_podataka})', fontsize=14)
    plt.xlabel('Prva značajka', fontsize=12)
    plt.ylabel('Druga značajka', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    X = generiraj_podatke(NUM_SAMPLES, DATA_TYPE)
    
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=RANDOM_STATE)
    kmeans.fit(X)
    
    labele = kmeans.labels_
    centri = kmeans.cluster_centers_
    
    vizualiziraj_rezultate(X, labele, centri, DATA_TYPE)

if __name__ == "__main__":
    main()
