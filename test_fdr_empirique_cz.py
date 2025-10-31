import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm




# extraction des CSV -- travail avec CMIP5 dans un premier temps (modèle plus simple)
obs = pd.read_csv("data/obs.csv", sep=';', parse_dates=["date"])
modele_hist = pd.read_csv("data/CMIP5_historical.csv", sep=';', parse_dates=["date"])
modele_fut = pd.read_csv("data/CMIP5_rcp85.csv", sep=';', parse_dates=["date"])

# ne garder qu'un mois, juillet pour cet exemple
jul_obs = obs[obs['date'].dt.month == 7]
jul_mod_hist = modele_hist[modele_hist['date'].dt.month == 7]
jul_mod_fut = modele_fut[modele_fut['date'].dt.month == 7]



# fonction de répartition empirique des températures pour un mois donné 
def ecdf(data):
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
def plot_ecdf_with_gaussian(data, show_gaussian=True):
    """
    Trace l'ECDF des données et, en option, la courbe d'une loi normale
    avec la même moyenne et écart-type que les données.
    """
    x, y = ecdf(data)
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



# calcul des ECDFs
#plot_ecdf_with_gaussian(jul_obs['tas'], show_gaussian=True)
#lot_ecdf_with_gaussian(jul_mod_hist['tas'], show_gaussian=True)
plot_ecdf_with_gaussian(jul_mod_fut['tas'], show_gaussian=True)
