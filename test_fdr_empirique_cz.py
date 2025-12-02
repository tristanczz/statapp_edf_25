import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gaussian_kde




# extraction des CSV -- travail avec CMIP5 dans un premier temps (modèle plus simple)
obs = pd.read_csv("data/obs.csv", sep=';', parse_dates=["date"])
modele_hist = pd.read_csv("data/CMIP5_historical.csv", sep=';', parse_dates=["date"])
modele_fut = pd.read_csv("data/CMIP5_rcp85.csv", sep=';', parse_dates=["date"])

# ne garder qu'un mois, juillet pour cet exemple
jul_obs = obs[obs['date'].dt.month == 7]
jul_mod_hist = modele_hist[modele_hist['date'].dt.month == 7]
jul_mod_fut = modele_fut[modele_fut['date'].dt.month == 7]



# fonction de répartition empirique des températures pour un mois donné 
def fdr(data):
    """Retourne l'ECDF (empirical cumulative distributive function) pour un ensemble de données
    ==> une liste de n couples (x, y) qui sont les coordonnées de la FDR empirique.
    ----------
    Paramètre :
    data : une liste de valeurs numériques(de témpratures ici, extraites du CSV) 
    ----------
    1) on calcule x
        Les valeurs de `data` triées par ordre croissant = les températures.
    2) on calcule y
        Les valeurs de la fonction de répartition empirique, 
        c'est-à-dire la proportion de données inférieures ou égales chaque valeur de `x`.
        y va de 1/n à 1.
    ------------
    Explication :
    Pour chaque valeur de x[i], y[i] correspond à la fraction des données 
    qui sont inférieures ou égales à x[i]. 
    Cela permet de visualiser la distribution cumulative des données.
    """
    x = np.sort(data)
    y = np.arange(1, len(x)+1) / len(x)
    return x, y




# fonction qui trace la FDR empirique (pour n points donc) et la compare à une loi normale 
# (avec paramètres mu = moyenne empirique et sigma^2 = variance empirique)
def graph_fdr(data, show_gaussian=True):
    """
    Trace l'ECDF des données et, en option, la courbe d'une loi normale
    avec la même moyenne et écart-type que les données.
    """
    x, y = fdr(data)
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker='.', linestyle='none', label="ECDF")
    
    if show_gaussian:
        mu, sigma = np.mean(data), np.std(data)
        x_gauss = np.linspace(min(data), max(data), 1000)
        y_gauss = norm.cdf(x_gauss, loc=mu, scale=sigma)
        plt.plot(x_gauss, y_gauss, color='red', label=f"Gaussienne N({mu:.2f}, {sigma:.2f})")
    plt.xlabel("Valeurs")
    plt.ylabel("Probabilité cumulée")
    plt.title("ECDF avec comparaison à une loi normale")
    plt.legend()
    plt.grid(True)
    plt.show()



#calcule la densité empirique à partir de la moyenne empirique et de la variance empirique
# sorte d'histogramme mais plus précis avec la KDE (Kernel Density Estimation)
def dens_empirique(data, n_points=10000):
    """
    Retourne la densité empirique (PDF empirique) d'un ensemble de données.
    ----------
    Paramètre :
    data : array-like
        Liste de valeurs numériques (par ex. températures).
    n_points : int
        Nombre de points pour l'échantillonnage de la densité.
    ----------
    Retour :
    x : ndarray
        Grille de valeurs triées.
    y : ndarray
        Densité empirique estimée (PDF empirique).
    """
    data = np.asarray(data)
    kde = gaussian_kde(data)  # estimation de densité à noyau
    x = np.linspace(min(data), max(data), n_points)
    y = kde(x)
    return x, y



def graph_dens(data, show_gaussian=True):
    """
    Trace la densité empirique (PDF empirique) et, en option,
    la courbe de densité d'une loi normale ajustée (même moyenne et variance).
    """
    x_emp, y_emp = dens_empirique(data)
    
    plt.figure(figsize=(8, 5))
    plt.plot(x_emp, y_emp, label="Densité empirique", color="blue")

    if show_gaussian:
        mu, sigma = np.mean(data), np.std(data)
        x_gauss = np.linspace(min(data), max(data), 1000)
        y_gauss = norm.pdf(x_gauss, loc=mu, scale=sigma)
        plt.plot(x_gauss, y_gauss, color='red', linestyle='--',
                 label=f"Gaussienne ajustée N({mu:.2f}, {sigma:.2f})")

    plt.xlabel("Valeurs")
    plt.ylabel("Densité de probabilité")
    plt.title("Densité empirique vs loi normale ajustée")
    plt.legend()
    plt.grid(True)
    plt.show()


graph_dens(jul_obs['tas'])