#dans le terminal : pip install pandas matplotlib statsmodels

import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def calculer_ecdf(nom_fichier, variable, annee_debut, annee_fin, mois=None):
    """
    Calcule l'ECDF d'une variable dans un fichier CSV pour une plage d'années et éventuellement un mois donné.
    """
    df = pd.read_csv(f"data/{nom_fichier}.csv", sep=";")
    df.columns = df.columns.str.strip()  # enlever espaces éventuels
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    # Filtrer années
    df = df[(df["date"].dt.year >= annee_debut) & (df["date"].dt.year <= annee_fin)]
    
    # Filtrer mois si spécifié
    if mois is not None:
        df = df[df["date"].dt.month == mois]
    
    valeurs = df[variable].dropna().values
    ecdf = ECDF(valeurs)
    return ecdf

def inverse_ecdf(ecdf, quantiles):
    """Calcule l'inverse empirique d'une ECDF : F^-1(q)"""
    return np.interp(quantiles, ecdf.y, ecdf.x)

def cdf_t(variable, donnees_hist_modele, donnees_hist_obs, donnees_futures_modele, annee_pass_debut, annee_pass_fin, annee_fut_debut, annee_fut_fin, mois=None):
    """
    Implémente la méthode CDF-t pour une variable donnée avec KDE lisse,
    en option pour un mois spécifique.
    F_f,r = F_p,r ∘ F_p,m^-1 ∘ F_f,m
    """
    # 1️⃣ ECDFs
    F_p_m = calculer_ecdf(donnees_hist_modele, variable, annee_pass_debut, annee_pass_fin, mois)
    F_p_r = calculer_ecdf(donnees_hist_obs, variable, annee_pass_debut, annee_pass_fin, mois)
    F_f_m = calculer_ecdf(donnees_futures_modele, variable, annee_fut_debut, annee_fut_fin, mois)

    # 2️⃣ CDF-t corrigée
    # Filtrage NaN avant KDE
    tas_f_m = F_f_m.x
    p_f_m = F_f_m.y
    tas_corrigees = inverse_ecdf(F_p_m, p_f_m)
    F_f_r = np.interp(tas_corrigees, F_p_r.x, F_p_r.y)

    # Supprimer NaN et inf
    tas_f_m = tas_f_m[~np.isnan(tas_f_m) & ~np.isinf(tas_f_m)]
    tas_corrigees = tas_corrigees[~np.isnan(tas_corrigees) & ~np.isinf(tas_corrigees)]

    if len(tas_f_m) == 0 or len(tas_corrigees) == 0:
        raise ValueError("Aucune donnée disponible pour le mois sélectionné et la période donnée.")

    # 3️⃣ Estimation KDE pour lisser les densités
    kde_f_m = gaussian_kde(tas_f_m)
    kde_f_r = gaussian_kde(tas_corrigees)

    x_grid = np.linspace(min(tas_f_m.min(), tas_corrigees.min()),
                         max(tas_f_m.max(), tas_corrigees.max()), 500)

    dens_f_m = kde_f_m(x_grid)
    dens_f_r = kde_f_r(x_grid)

    # 4️⃣ Tracé
    plt.figure(figsize=(8,5))
    plt.plot(x_grid, dens_f_m, label="Modèle futur", color="red")
    plt.plot(x_grid, dens_f_r, label="Futur corrigé (CDF-t)", color="blue")
    plt.xlabel(variable)
    plt.ylabel("Densité KDE")
    titre_mois = f" - Mois {mois}" if mois is not None else ""
    plt.title(f"Densités futures (KDE lisse) : {variable}{titre_mois}")
    plt.legend()
    plt.grid(True)
    plt.show()

    return x_grid, dens_f_m, dens_f_r


#CMIP6 sur obs
cdf_t("tas","CMIP6_historical","obs","CMIP6_ssp585",1995,2014,2080,2099, mois =7)

#CMIP5 sur Eurocordex
cdf_t("tas","CMIP5_historical","eurocordex_historical","CMIP5_rcp85",1995,2014,2080,2099, mois =7)





