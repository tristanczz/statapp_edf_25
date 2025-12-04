import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gaussian_kde
from fdr_empirique import fdr, graph_fdr, graph_fdr_12, cdf_t_correction




# extraction des CSV -- travail avec CMIP5 dans un premier temps (modèle plus simple)
obs = pd.read_csv("data/obs.csv", sep=';', parse_dates=["date"])
obs.insert(obs.columns.get_loc('tas') + 1, 'sfcWind', np.sqrt(obs['uas']**2 + obs['vas']**2))  # Ajout de la colonne 'sfcWind' après 'tas'
modele_hist = pd.read_csv("data/CMIP5_historical.csv", sep=';', parse_dates=["date"])
modele_fut = pd.read_csv("data/CMIP5_rcp85.csv", sep=';', parse_dates=["date"])


# Filtrage par mois  -- exemple pour Juillet
jul_obs = obs[obs['date'].dt.month == 7]
jul_mod_hist = modele_hist[modele_hist['date'].dt.month == 7]
jul_mod_fut = modele_fut[modele_fut['date'].dt.month == 7]



# Boucle des mois  -- Dictionnaires
obs_mois = {}
modele_hist_mois = {}
modele_fut_mois = {}
for i in range(1, 13):
    obs_mois[i] = obs[obs['date'].dt.month == i]
    modele_hist_mois[i] = modele_hist[modele_hist['date'].dt.month == i]
    modele_fut_mois[i] = modele_fut[modele_fut['date'].dt.month == i]


# FDR par mois 
plt.figure(figsize=(10,6))
ax = plt.gca()
couleurs = plt.cm.tab20.colors  # 20 couleurs possibles
for i in range(1, 13):
    tas_mois = obs_mois[i]['tas']  # <- on prend uniquement la colonne des températures
    graph_fdr_12(tas_mois, ax=ax, show_gaussian=False, label=f"Mois {i}", couleur=couleurs[i-1])
ax.legend()
#plt.title("FDR par mois")
#plt.xlabel("Valeurs")
#plt.ylabel("Probabilité cumulée")
#plt.show()


# FDR globale
# A faire


# Prevision s futures / Corrections CDFT
print(cdf_t_correction(jul_mod_hist, jul_obs, jul_mod_fut))



