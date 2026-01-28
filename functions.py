import pandas as pd
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


def csv_to_pd_univ(filepath, mois, var, years=None):
    """
    Charge un CSV à partir d'un chemin donné,
    filtre pour ne garder que les valeurs de la variable 'var' pour le mois indiqué
    et optionnellement pour une période d'années.

    Paramètres
    ----------
    filepath : str
        Chemin complet vers le fichier CSV (ex: '/home/onyxia/work/statapp_edf_25/data/obs.csv')
    mois : int
        Mois à sélectionner (1=Janvier, 12=Décembre)
    var : str
        Nom de la variable à conserver
    years : tuple, optional
        Tuple (année_début, année_fin) pour filtrer la période
        Ex: (1985, 2005) pour extraire les données de 1985 à 2005 inclus

    Retour
    ------
    pd.DataFrame
        DataFrame contenant uniquement la colonne 'var' filtrée sur le mois et la période demandés.
    """
    if years is not None and (not isinstance(years, tuple) or len(years) != 2):
        raise ValueError(f"years doit être un tuple de 2 éléments (année_début, année_fin)")
    
    # Lire le CSV avec ; et parser la colonne date
    df = pd.read_csv(filepath, sep=';', parse_dates=['date'])

    # Filtrer sur le mois
    df = df[df['date'].dt.month == mois]
    
    # Filtrer sur la période d'années si spécifié
    if years is not None:
        year_start, year_end = years
        df = df[(df['date'].dt.year >= year_start) & (df['date'].dt.year <= year_end)]
    
    # Vérifier qu'il reste des données
    if len(df) == 0:
        period_str = f" pour la période {years[0]}-{years[1]}" if years else ""
        raise ValueError(f"Aucune donnée trouvée pour le mois {mois} et la variable {var}{period_str}")

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
    ecdf_modele_hist = ECDF(m_hist)

    # Étape 1:
    p_futur = ecdf_modele_futur(m_futur)

    # Étape 2:
    val_hist = np.quantile(o_hist, p_futur)

    # Étape 3:
    p_obs = ecdf_modele_hist(val_hist)

    # Étape 4:
    obs_futur_corrige = np.quantile(m_futur, p_obs)

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

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
    plt.close('all')


def comparer_distributions(df):
    """
    Compare les statistiques descriptives de deux distributions.

    Paramètres:
    -----------
    df : DataFrame
        DataFrame contenant deux colonnes à comparer

    Retourne:
    ---------
    DataFrame
        Tableau comparatif des statistiques
    """
    # Récupérer les noms des colonnes
    col1, col2 = df.columns[0], df.columns[1]

    # Calculer les statistiques pour chaque colonne
    stats = {
        'Statistique': ['Moyenne', 'Médiane', 'Écart-type', 'Variance',
                        'Minimum', 'Maximum', 'Q1 (25%)', 'Q3 (75%)',
                        'Étendue',],
        col1: [
            df[col1].mean(),
            df[col1].median(),
            df[col1].std(),
            df[col1].var(),
            df[col1].min(),
            df[col1].max(),
            df[col1].quantile(0.25),
            df[col1].quantile(0.75),
            df[col1].max() - df[col1].min(),
        ],
        col2: [
            df[col2].mean(),
            df[col2].median(),
            df[col2].std(),
            df[col2].var(),
            df[col2].min(),
            df[col2].max(),
            df[col2].quantile(0.25),
            df[col2].quantile(0.75),
            df[col2].max() - df[col2].min(),
        ]
    }

    # Créer le DataFrame de comparaison
    df_stats = pd.DataFrame(stats)

    # Arrondir les valeurs
    df_stats = df_stats.round(3)

    return df_stats


def cdf_t_univ_norm(modele_hist, obs_hist, modele_futur):

    # Extraire les valeurs des colonnes
    m_hist = modele_hist.iloc[:, 0].values
    o_hist = obs_hist.iloc[:, 0].values
    m_futur = modele_futur.iloc[:, 0].values

    # Estimation des paramètres des gaussiennes par maximum de vraisemblance
    mu_obs_hist, sigma_obs_hist = norm.fit(o_hist)
    mu_model_hist, sigma_model_hist = norm.fit(m_hist)
    mu_model_fut, sigma_model_fut = norm.fit(m_futur)

    # Calcul du paramètre de la loi (normale) des observations futures
    mu_obs_fut = mu_model_fut + (mu_obs_hist-mu_model_hist)*(sigma_model_fut/sigma_model_hist)
    sigma_obs_fut = sigma_obs_hist*(sigma_model_fut/sigma_model_hist)

    return [(mu_model_fut, sigma_model_fut), (mu_obs_fut, sigma_obs_fut)]


def plot_distributions_univ_norm(norm_model, norm_obs, nom_modele, mois, var):
    """
    Trace deux distributions normales avec leurs paramètres.

    Paramètres:
    -----------
    norm_model : tuple (mean, std)
        Paramètres de la loi normale du modèle (moyenne, écart-type)
    norm_obs : tuple (mean, std)
        Paramètres de la loi normale observée (moyenne, écart-type)
    nom_modele : str
        Nom du modèle
    var : str
        Nom de la variable
    mois : int
        Numéro du mois (1-12)
    """
    # Noms des mois
    mois_noms = {
        1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
        5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
        9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
    }

    # Décomposer les paramètres
    mean_model, std_model = norm_model
    mean_obs, std_obs = norm_obs

    # Créer une plage de valeurs pour tracer les courbes
    x_min = min(mean_model - 4*std_model, mean_obs - 4*std_obs)
    x_max = max(mean_model + 4*std_model, mean_obs + 4*std_obs)
    x = np.linspace(x_min, x_max, 1000)

    # Calculer les densités
    y_model = norm.pdf(x, mean_model, std_model)
    y_obs = norm.pdf(x, mean_obs, std_obs)

    # Configurer le style
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    # Tracer les courbes
    plt.plot(x, y_model, linewidth=2.5, color='#e74c3c', alpha=0.8,
             label=f'Modèle: μ={mean_model:.2f}, σ={std_model:.2f}')
    plt.plot(x, y_obs, linewidth=2.5, color='#3498db', alpha=0.8,
             label=f'Corrigé: μ={mean_obs:.2f}, σ={std_obs:.2f}')

    # Ajouter des zones ombrées sous les courbes
    plt.fill_between(x, y_model, alpha=0.2, color='#e74c3c')
    plt.fill_between(x, y_obs, alpha=0.2, color='#3498db')

    # Labels et titre
    plt.xlabel('Valeur', fontsize=12)
    plt.ylabel('Densité de probabilité', fontsize=12)
    plt.title(f'Correction du modèle {nom_modele} pour {var} et {mois_noms[mois]}', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def extract_month_data(csv_path, month, columns=None, years=None):
    """
    Extrait les données d'un mois spécifique d'un fichier CSV pour une période donnée.
    
    Parameters:
    -----------
    csv_path : str
        Chemin vers le fichier CSV (séparateur ';')
    month : int
        Numéro du mois à extraire (1-12)
    columns : list, optional
        Liste des colonnes à extraire (dans l'ordre souhaité)
    years : tuple, optional
        Tuple (année_début, année_fin) pour filtrer la période
        Ex: (1985, 2005) pour extraire les données de 1985 à 2005 inclus
    
    Returns:
    --------
    numpy.ndarray
        Array de shape (n_samples, n_variables) avec les données du mois et de la période
    """
    if not 1 <= month <= 12:
        raise ValueError(f"Le mois doit être entre 1 et 12, reçu: {month}")
    
    if years is not None and (not isinstance(years, tuple) or len(years) != 2):
        raise ValueError(f"years doit être un tuple de 2 éléments (année_début, année_fin)")
    
    df = pd.read_csv(csv_path, sep=';')
    date_col = df.columns[0]
    
    # Conversion flexible pour gérer différents formats de date
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Filtrer par mois
    df_filtered = df[df[date_col].dt.month == month]
    
    # Filtrer par période d'années si spécifié
    if years is not None:
        year_start, year_end = years
        df_filtered = df_filtered[
            (df_filtered[date_col].dt.year >= year_start) & 
            (df_filtered[date_col].dt.year <= year_end)
        ]
    
    if len(df_filtered) == 0:
        period_str = f" pour la période {years[0]}-{years[1]}" if years else ""
        raise ValueError(f"Aucune donnée trouvée pour le mois {month}{period_str}")
    
    # Si des colonnes spécifiques sont demandées
    if columns is not None:
        data = df_filtered[columns].values
    else:
        data = df_filtered.iloc[:, 1:].values
    
    period_info = f" ({years[0]}-{years[1]})" if years else ""
    print(f"Données extraites de {csv_path}: {len(data)} lignes, {data.shape[1]} variables pour le mois {month}{period_info}")
    
    return data