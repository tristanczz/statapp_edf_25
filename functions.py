import pandas as pd
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import seaborn as sns


def csv_to_pd_univ(filepath, mois, var):
    """
    Charge un CSV à partir d'un chemin donné,
    filtre pour ne garder que les valeurs de la variable 'var' pour le mois indiqué.

    Paramètres
    ----------
    filepath : str
        Chemin complet vers le fichier CSV (ex: '/home/onyxia/work/statapp_edf_25/data/obs.csv')
    mois : int
        Mois à sélectionner (1=Janvier, 12=Décembre)
    var : str
        Nom de la variable à conserver

    Retour
    ------
    pd.DataFrame
        DataFrame contenant uniquement la colonne 'var' filtrée sur le mois demandé.
    """
    # Lire le CSV avec ; et parser la colonne date
    df = pd.read_csv(filepath, sep=';', parse_dates=['date'])

    # Filtrer sur le mois
    df = df[df['date'].dt.month == mois]

    # Ne garder que la colonne var
    df_final = df[[var]].copy()

    return df_final


def cdf_t_univ(modele_hist, obs_hist, modele_futur):
    """
    Applique la méthode CDF-t pour corriger les biais d'un modèle climatique.

    Paramètres:
    -----------
    modele_hist : DataFrame
        Données historiques du modèle (une seule colonne)
    obs_hist : DataFrame
        Observations historiques (une seule colonne)
    modele_futur : DataFrame
        Projections futures du modèle (une seule colonne)

    Retourne:
    ---------
    DataFrame
        DataFrame avec deux colonnes : 'valeurs_modele' et 'valeurs_corrigees'
    """
    # Extraire les valeurs des colonnes
    m_hist = modele_hist.iloc[:, 0].values
    o_hist = obs_hist.iloc[:, 0].values
    m_futur = modele_futur.iloc[:, 0].values

    # Calculer les fonctions de répartition empiriques (ECDF)
    ecdf_modele_futur = ECDF(m_futur)
    ecdf_obs_hist = ECDF(o_hist)

    # Vectorisation : appliquer les transformations sur tout le vecteur
    # Étape 1: ecdf_modele_futur pour toutes les valeurs futures
    p_futur = ecdf_modele_futur(m_futur)

    # Étape 2: ecdf_modele_hist^-1 = quantiles de modele_hist
    val_hist = np.quantile(m_hist, p_futur)

    # Étape 3: ecdf_obs_hist sur toutes les valeurs historiques correspondantes
    p_obs = ecdf_obs_hist(val_hist)

    # Étape 4: Inverser = quantiles de obs_hist
    obs_futur_corrige = np.quantile(o_hist, p_obs)

    # Retourner un DataFrame avec deux colonnes
    return pd.DataFrame({
        'valeurs_modele': m_futur,
        'valeurs_corrigees': obs_futur_corrige
    }, index=modele_futur.index)


def plot_distributions_univ(df, nom_modele, mois, var):
    """
    Trace les distributions de deux colonnes d'un DataFrame en les superposant.

    Paramètres:
    -----------
    df : DataFrame
        DataFrame contenant deux colonnes à comparer
    nom_modele : str
        Nom du modèle pour le titre

    Retourne:
    ---------
    None (affiche le graphique)
    """
    # Configurer le style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))

    # Récupérer les noms des colonnes
    col1, col2 = df.columns[0], df.columns[1]

    # Créer deux sous-graphiques
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Graphique 1 : Courbes de densité (KDE)
    sns.kdeplot(data=df[col1], label=col1, linewidth=2.5, ax=ax1, color='#e74c3c', alpha=0.7)
    sns.kdeplot(data=df[col2], label=col2, linewidth=2.5, ax=ax1, color='#3498db', alpha=0.7)
    ax1.set_xlabel('Valeur', fontsize=11)
    ax1.set_ylabel('Densité', fontsize=11)
    ax1.set_title('Distributions (courbes de densité)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Graphique 2 : Histogrammes superposés
    ax2.hist(df[col1], bins=30, alpha=0.5, label=col1, color='#e74c3c', density=True, edgecolor='black')  # noqa: E501
    ax2.hist(df[col2], bins=30, alpha=0.5, label=col2, color='#3498db', density=True, edgecolor='black')  # noqa: E501
    ax2.set_xlabel('Valeur', fontsize=11)
    ax2.set_ylabel('Densité', fontsize=11)
    ax2.set_title('Distributions (histogrammes)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, axis='y')

    # Dictionnaire des mois pour le titre
    dic_mois = {1: "janvier", 2: "février", 3: "mars", 4: "avril", 5: "mai", 6: "juin",
                7: "juillet", 8: "août", 9: "septembre", 10: "octobre",
                11: "novembre", 12: "décembre"}

    # Titre général
    fig.suptitle(
        f"Correction du modèle {nom_modele} pour {var} et {dic_mois[mois]}", fontsize=14,
        fontweight='bold', y=1.02
    )

    plt.tight_layout()
    plt.show()
