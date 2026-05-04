import pandas as pd
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import seaborn as sns
import ot  
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde

def csv_to_pd(filepath, mois=None, var=None, years=None):
    """
    Charge un CSV à partir d'un chemin donné,
    filtre pour ne garder que les valeurs des variables dans var pour le mois indiqué
    et optionnellement pour une période d'années.

    Paramètres
    ----------
    filepath : str
        Chemin complet vers le fichier CSV (ex: '/home/onyxia/work/statapp_edf_25/data/obs.csv')
    mois : int
        Mois à sélectionner (1=Janvier, 12=Décembre)
    var : list
        Liste des noms des variables à conserver
    years : tuple, optional
        Tuple (année_début, année_fin) pour filtrer la période
        Ex: (1985, 2005) pour extraire les données de 1985 à 2005 inclus

    Retour
    ------
    pd.DataFrame
        DataFrame contenant uniquement les colonnes 'var' filtrées sur le mois et la période demandés.
    """
    if years is not None and (not isinstance(years, tuple) or len(years) != 2):
        raise ValueError(f"years doit être un tuple de 2 éléments (année_début, année_fin)")
    
    # Lire le CSV avec ; et parser la colonne date
    df = pd.read_csv(filepath, sep=';', parse_dates=['date'])

    # Filtrer sur le mois
    if mois is not None:
        df = df[df['date'].dt.month == mois]
    
    # Filtrer sur la période d'années si spécifié
    if years is not None:
        year_start, year_end = years
        df = df[(df['date'].dt.year >= year_start) & (df['date'].dt.year <= year_end)]
    
    # Vérifier qu'il reste des données
    if len(df) == 0:
        period_str = f" pour la période {years[0]}-{years[1]}" if years else ""
        raise ValueError(f"Aucune donnée trouvée pour le mois {mois} et la variable {var}{period_str}")

    # Ne garder que les variables dans var 
    if var is not None:
        df_final = df[var].copy()
    else:
        df_final = df.copy()

    return df_final


def select_df(df, month=None, var=None, start_year=None, end_year=None):
    """
    Sélectionne les données d'un dataframe en fonction du mois, des variables et de la période d'années souhaités.)

    """

    result = df.copy()

    if month is not None:
        result = result[result['month'] == month]
    
    if start_year is not None and end_year is not None:
        result = result[(result['year'] >= start_year) & (result['year'] <= end_year)]

    if var is not None:
        result = result[var]
    
    return result



def cdf_t_univ(modele_hist, obs_hist, modele_futur):
    """
    Applique la méthode CDF-t pour corriger les biais d'un modèle climatique.

    Paramètres:
    -----------
    modele_hist : Numpy array
        Données historiques du modèle (une seule colonne)
    obs_hist : Numpy array
        Observations historiques (une seule colonne)
    modele_futur : Numpy array
        Projections futures du modèle (une seule colonne)

    Retourne:
    ---------
    modele_non_biaise : pd.DataFrame
        Données futures corrigées avec les dates associées
    ecdf_non_biaise : function
        Fonction de répartition empirique du modèle corrigé
    """

    # Fonctions de répartition empiriques 
    ecdf_modele_futur = ECDF(modele_futur) #F_X1
    ecdf_modele_hist = ECDF(modele_hist) # F_X0

    
    #Quantile mapping en supposant que la relation de stationnarité est vérifiée :
     
    q_futur = ecdf_modele_futur(modele_futur) # F_X1(X1)

    b = np.quantile(obs_hist, q_futur) #F_Y0^{-1}(F_X1(X1))

    c = ecdf_modele_hist(b) #F_X0(b)

    modele_non_biaise = np.quantile(modele_futur, c) #F_Y1^{-1}(c)

    # ECDF du modele non biaisé
    def ecdf_non_biaise(x):
        
        p = ecdf_modele_futur(x)
        b = np.quantile(modele_hist, p)
        return ECDF(obs_hist)(b)

    
    return modele_non_biaise, ecdf_non_biaise



def cdf_t_multivar(modele_hist, obs_hist, modele_futur):
    """
    Applique la méthode CDF-t variable par variable (quantile mapping multivarié naïf).
    
    Paramètres:
    -----------
    modele_hist : Numpy array (n_samples, n_features)
        Données historiques du modèle
    obs_hist : Numpy array (n_samples, n_features)
        Observations historiques
    modele_futur : Numpy array (m_samples, n_features)
        Projections futures du modèle
    
    Retourne:
    ---------
    modele_non_biaise : Numpy array (m_samples, n_features)
        Données futures corrigées variable par variable
    """
    n_features = modele_hist.shape[1]
    n_samples = modele_futur.shape[0]
    
    modele_non_biaise = np.zeros((n_samples, n_features))
    
    # Appliquer CDF-t sur chaque variable indépendamment
    for d in range(n_features):
        modele_non_biaise[:, d], _ = cdf_t_univ(
            modele_hist[:, d],
            obs_hist[:, d],
            modele_futur[:, d]
        )
    
    return modele_non_biaise


def plot_year_per_year_crossed(obs, modele, month, var, first_year, last_year, method = 'cdf-t'):
    
    obs_past = obs[(obs['year'] < first_year) & (obs['month'].isin(month))][var].values

    modele_past = modele[(modele['year'] < first_year) & (modele['month'].isin(month))][var].values
    modele_fut = modele[(modele['year'] >= first_year) & (modele['year'] <= last_year) & (modele['month'].isin(month))][var].values

    modele_corrige, _ = cdf_t_univ(modele_past, obs_past, modele_fut)


    df = pd.DataFrame({
        'valeurs_modele': modele_fut,
        'valeurs_corrigees': modele_corrige,
        'year': modele[(modele['year'] >= first_year) & (modele['year'] <= last_year) & (modele['month'].isin(month))]['year'].values
    })

    per_year_obs_fut = [obs[obs['year'] == m][var].mean() for m in range(first_year, last_year+1)]
    per_year_modele_corrige = [df[df['year'] == m]['valeurs_corrigees'].mean() for m in range(first_year, last_year+1)]
    per_year_modele_fut = [modele[modele['year'] == m][var].mean() for m in range(first_year, last_year+1)]

    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.xlabel('Année')
    plt.ylabel(f'{var}')
    plt.plot(range(first_year, last_year+1), per_year_obs_fut, 'o-', label='Observations')
    plt.plot(range(first_year, last_year+1), per_year_modele_corrige, 'o-', label='Modèle CMIP6 corrigé')
    plt.plot(range(first_year, last_year+1), per_year_modele_fut, 'o-', label='Modèle CMIP6 non corrigé')

    plt.legend()
    plt.show()

    return df


def plot_year_per_year_fut(obs, modele, month, var):
    
    obs_past = obs[(obs['year'] < 2022) & (obs['month'].isin(month))][var].values

    modele_past = modele[(modele['year'] < 2022) & (modele['month'].isin(month))][var].values
    modele_fut = modele[(modele['year'] >= 2022) & (modele['year'] <= 2100) & (modele['month'].isin(month))][var].values

    modele_corrige, _ = cdf_t_univ(modele_past, obs_past, modele_fut)


    df = pd.DataFrame({
        'valeurs_modele': modele_fut,
        'valeurs_corrigees': modele_corrige,
        'year': modele[(modele['year'] >= 2022) & (modele['year'] <= 2100) & (modele['month'].isin(month))]['year'].values
    })

    per_year_modele_corrige = [df[df['year'] == m]['valeurs_corrigees'].mean() for m in range(2022, 2101)]
    per_year_modele_fut = [modele[modele['year'] == m][var].mean() for m in range(2022, 2101)]

    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.xlabel('Année')
    plt.ylabel(f'{var}')
    plt.plot(range(2022, 2101), per_year_modele_corrige, 'o-', label='Modèle CMIP6 corrigé')
    plt.plot(range(2022, 2101), per_year_modele_fut, 'o-', label='Modèle CMIP6 non corrigé')

    plt.legend()
    plt.show()

    return df
    


def plot_distributions(data, titre, labels_tuple):
    """
    Trace les distributions de deux DataFrames en les superposant.
    Histogrammes et courbes de densité sont combinés sur le même graphique.

    Paramètres:
    -----------
    dfs_tuple : tuple(DataFrame, DataFrame)
        Tuple contenant deux DataFrames (futur, passé) avec une seule colonne chacun
    titre : str

    Retourne:
    ---------
    None (affiche le graphique)
    """
    # Configurer le style
    sns.set_style("whitegrid")
    
    # Extraire les deux DataFrames du tuple
    data1, data2 = data

    #Récupérer les labels
    label1, label2 = labels_tuple

    # Créer la figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Histogrammes superposés
    ax.hist(data1, bins=30, alpha=0.4, label=f' {label1} (histogramme)', 
            color='#e74c3c', density=True, edgecolor='black', linewidth=0.5)
    ax.hist(data2, bins=30, alpha=0.4, label=f'{label2} (histogramme)', 
            color='#3498db', density=True, edgecolor='black', linewidth=0.5)

    # Courbes de densité par-dessus
    sns.kdeplot(data=data1, label=f' {label1} (densité)', linewidth=2.5, 
                ax=ax, color='#c0392b', alpha=0.9)
    sns.kdeplot(data=data2, label=f'{label2} (densité)', linewidth=2.5, 
                ax=ax, color='#2980b9', alpha=0.9)

    # Labels et titre
    ax.set_xlabel('Valeur', fontsize=12)
    ax.set_ylabel('Densité', fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.3, axis='y')

    # Titre général
    ax.set_title(
        titre, 
        fontsize=14, fontweight='bold', pad=15
    )

    plt.tight_layout()
    plt.show()
    plt.close('all')


def comparer_distributions_univ(dis1, dis2):
    """
    Compare les statistiques descriptives de deux distributions.

    Paramètres:
    
    dfs_tuple : Numpy arrays
        Tuple contenant deux DataFrames avec une seule colonne chacun

    Retourne:

    DataFrame
        Tableau comparatif des statistiques
    """


    # Calculer les statistiques pour chaque colonne
    stats = {
        'Statistique': ['Moyenne', 'Médiane', 'Écart-type', 'Variance',
                        'Minimum', 'Maximum', 'Q1 (25%)', 'Q3 (75%)',
                        'Étendue'],
        'Distribution 1': [
            dis1.mean(),
            np.median(dis1),
            np.std(dis1),
            np.var(dis1),
            dis1.min(),
            dis1.max(),
            np.quantile(dis1, 0.25),
            np.quantile(dis1, 0.75),
            dis1.max() - dis1.min(),
        ],
        'Distribution 2': [
            dis2.mean(),
            np.median(dis2),
            np.std(dis2),
            np.var(dis2),
            dis2.min(),
            dis2.max(),
            np.quantile(dis2, 0.25),
            np.quantile(dis2, 0.75),
            dis2.max() - dis2.min(),
        ]
    }

    df_stats = pd.DataFrame(stats)
    df_stats = df_stats.round(3)

    return df_stats



def transport_optimal(X0, Y0, X1, weight='uniform', method='emd', reg=0.01, translation=False, normalize=True):
    """
    Paramètres:

    X0 : array (n0, d)
        modèle dans le passé
    X1 : array (n1, d)
        modèle futur à corriger
    Y0 : array (m, d)
        observations
    method : str, default='emd'
        'emd' pour transport exact (lent, sparse)
        'sinkhorn' pour transport régularisé (rapide, moins sparse)
    reg : float, default=0.01
        Paramètre de régularisation pour sinkhorn
    normalize : bool, default=True
        Normaliser les variables avant de calculer le transport (recommandé pour données multivariées)

    Returns:

    Y1 : array (m, d)
        modèle corrigée
    info : dict
        Dictionnaire contenant:
        - 'T_matrix': matrice de transport (n0, n1)
        - 'mapping': array (n0,) indices de correspondance X0 → X1
        - 'cost': coût total du transport
        - 'weights_source': poids a utilisés
        - 'weights_target': poids b utilisés
    """
    n0, d = X0.shape
    n1 = X1.shape[0]
    m = Y0.shape[0]

    # Normalisation par variable pour éviter que les variables à grande échelle dominent le transport
    if normalize:
        mu = X0.mean(axis=0)
        sigma = X0.std(axis=0) + 1e-8
        X0n = (X0 - mu) / sigma
        Y0n = (Y0 - mu) / sigma
        X1n = (X1 - mu) / sigma
    else:
        X0n, Y0n, X1n = X0, Y0, X1

    if weight == 'uniform':
        a = np.ones(n0) / n0
        b = np.ones(n1) / n1
    else:
        kde_X0 = gaussian_kde(X0n.T, bw_method='scott')
        densities_X0 = kde_X0(X0n.T)
        kde_X1 = gaussian_kde(X1n.T, bw_method='scott')
        densities_X1 = kde_X1(X1n.T)
        a = densities_X0 / densities_X0.sum()
        b = densities_X1 / densities_X1.sum()

    M = ot.dist(X0n, X1n, metric='euclidean')

    if method == 'emd':
        T_matrix = ot.emd(a, b, M)
    elif method == 'sinkhorn':
        T_matrix = ot.sinkhorn(a, b, M, reg)

    cost = np.sum(T_matrix * M)
    nnz = np.sum(T_matrix > 1e-10)
    sparsity = 1 - (nnz / T_matrix.size)

    # Pour chaque Y0[i], trouver le plus proche voisin dans X0 (espace normalisé)
    distances = cdist(Y0n, X0n, metric='euclidean')
    nearest_in_X0 = np.argmin(distances, axis=1)

    if method == 'sinkhorn':
        # Projection barycentrique douce : moyenne pondérée des X1 selon les poids de transport
        row_weights = T_matrix[nearest_in_X0]          # (m, n1)
        row_weights = row_weights / row_weights.sum(axis=1, keepdims=True)
        Y1n = row_weights @ X1n                         # (m, d)
        mapping = np.argmax(T_matrix, axis=1)
        if translation:
            X0_nn = X0n[nearest_in_X0]
            Y1n = Y0n + (Y1n - X0_nn)
    else:
        mapping = np.argmax(T_matrix, axis=1)
        indices_Y1 = mapping[nearest_in_X0]
        if translation:
            Y1n = Y0n + (X1n[indices_Y1] - X0n[nearest_in_X0])
        else:
            Y1n = X1n[indices_Y1]

    # Dénormalisation
    if normalize:
        Y1 = Y1n * sigma + mu
    else:
        Y1 = Y1n

    info = {
        'T_matrix': T_matrix,
        'mapping': mapping,
        'cost': cost,
        'weights_source': a,
        'weights_target': b,
        'nearest_in_X0': nearest_in_X0,
        'method': method,
        'sparsity': sparsity
    }

    return Y1, info


def schrodinger_bridge(X0, Y0, X1, reg=0.5, normalize=True, translation=False):
    """
    Correction de biais par pont de Schrödinger : transport optimal régularisé (Sinkhorn)
    avec projection barycentrique douce.

    Paramètres:
    -----------
    X0 : array (n0, d) - modèle dans le passé
    Y0 : array (m, d)  - observations dans le passé
    X1 : array (n1, d) - modèle futur à corriger
    reg : float        - régularisation Sinkhorn (plus grand = plus diffus)
    normalize : bool   - normaliser les variables avant le transport
    translation : bool - mode delta : Y1 = Y0 + (barycentre - X0_voisin)

    Returns:
    --------
    Y1 : array (m, d) - projections futures corrigées
    info : dict
    """
    n0, d = X0.shape
    n1 = X1.shape[0]
    m = Y0.shape[0]

    if normalize:
        mu = X0.mean(axis=0)
        sigma = X0.std(axis=0) + 1e-8
        X0n = (X0 - mu) / sigma
        Y0n = (Y0 - mu) / sigma
        X1n = (X1 - mu) / sigma
    else:
        X0n, Y0n, X1n = X0, Y0, X1

    a = np.ones(n0) / n0
    b = np.ones(n1) / n1

    # Matrice de coût et plan de transport Sinkhorn (pont de Schrödinger)
    M = ot.dist(X0n, X1n, metric='sqeuclidean')
    T = ot.sinkhorn(a, b, M, reg)

    # Pour chaque Y0[i], trouver le voisin le plus proche dans X0
    nn_idx = np.argmin(cdist(Y0n, X0n, metric='euclidean'), axis=1)  # (m,)

    # Projection barycentrique douce via la ligne de transport du voisin X0
    row_weights = T[nn_idx]                                            # (m, n1)
    row_weights = row_weights / row_weights.sum(axis=1, keepdims=True)
    bary_n = row_weights @ X1n                                         # (m, d)

    if translation:
        Y1n = Y0n + (bary_n - X0n[nn_idx])
    else:
        Y1n = bary_n

    if normalize:
        Y1 = Y1n * sigma + mu
    else:
        Y1 = Y1n

    info = {
        'T_matrix': T,
        'reg': reg,
        'normalize': normalize,
        'nn_idx': nn_idx,
    }

    return Y1, info



def prediction_score(Y0, Y1):
    """
    Compare deux distributions multivariées.
    
    Paramètres:

    Y0 : array (m, d)
        observations
    Y1 : array (n, d)
        Distribution corrigée

    Retourne:

    frobenius_cov : float
        Distance de Frobenius entre matrices de covariance
    wasserstein : float
        Distance de Wasserstein 2D
    score : float
        Score de qualité entre 0 (mauvais) et 1 (parfait)
    """

    # distance de Frobenius entre les matrices de covariance
    cov_corrected = np.cov(Y1.T)
    cov_target = np.cov(Y0.T)
    frobenius_cov = np.linalg.norm(cov_corrected - cov_target, 'fro')
    

    # distance de Wasserstein entre les distributions empiriques, ou coût du transport optimal 
    a = np.ones(len(Y1)) / len(Y1)
    b = np.ones(len(Y0)) / len(Y0)
    M = ot.dist(Y1, Y0, metric='euclidean')
    wasserstein = ot.emd2(a, b, M) 
    
    # normalisation pour obtenir des scores entre 0 et 1
    cov_norm = np.linalg.norm(cov_target, 'fro')
    score_cov = max(0, 1 - frobenius_cov / (cov_norm + 1e-10))
    
    baseline_wass = np.trace(cov_target)
    score_wass = max(0, 1 - wasserstein / (baseline_wass + 1e-10))
    
    # combinaison des scores
    score = 0.5 * score_cov + 0.5 * score_wass
    
    return frobenius_cov, wasserstein, score

def cov_score(Y0, Y1):
    """
    Calcule un score de qualité basé uniquement sur la covariance

    Paramètres:

    Y0 : array (m, d)
        observations
    Y1 : array (n, d)
        Distribution corrigée

    Retourne:

    score_cov : float
        Score de qualité entre 0 (mauvais) et 1 (parfait) basé sur la covariance
    """
    cov_corrected = np.cov(Y1.T)
    cov_target = np.cov(Y0.T)

    if Y0.shape[1] == 1 :
        frobenius_cov = np.abs(cov_corrected - cov_target)
        cov_norm = np.abs(cov_target)

    else :
        frobenius_cov = np.linalg.norm(cov_corrected - cov_target, 'fro')
        cov_norm = np.linalg.norm(cov_target, 'fro')

    score_cov = max(0, 1 - frobenius_cov / (cov_norm + 1e-10))

    return frobenius_cov, score_cov


def correlation_frobenius(Y0, Y1):
    """
    Erreur de Frobenius entre matrices de corrélation (sans dimension, robuste aux différences d'échelle).

    Paramètres:
    -----------
    Y0 : array (m, d) - distribution cible (observations)
    Y1 : array (n, d) - distribution corrigée

    Retourne:
    ---------
    frob : float - ||R_Y1 - R_Y0||_F
    """
    if Y0.shape[1] == 1:
        return 0.0
    R0 = np.corrcoef(Y0.T)
    R1 = np.corrcoef(Y1.T)
    return float(np.linalg.norm(R1 - R0, 'fro'))


def gaussian_multivar_test(d, N=1000, N_fut=500, reg_sb=0.5, seed=42):
    """
    Test contrôlé sur données gaussiennes multivariées.
    Génère X0 (modèle passé), Y0 (obs passé), X1 (modèle futur),
    applique CDF-t, OT, SB, et compare les distributions corrigées à Y0.

    Paramètres:
    -----------
    d     : int   - nombre de variables
    N     : int   - taille des échantillons passés
    N_fut : int   - taille de l'échantillon futur
    reg_sb: float - régularisation pour le pont de Schrödinger
    seed  : int

    Retourne:
    ---------
    results : dict avec clés 'cdf_t', 'ot', 'sb'
        Chaque valeur est un dict: {'frobenius_corr', 'wasserstein', 'mean_error'}
    meta : dict - paramètres théoriques et données générées
    """
    rng = np.random.default_rng(seed)

    # Matrices de covariance aléatoires définies positives
    A = rng.standard_normal((d, d))
    Sigma_y0 = A @ A.T / d + np.eye(d)

    B = rng.standard_normal((d, d))
    Sigma_x0 = B @ B.T / d + np.eye(d)

    # Signal de changement climatique : décalage en moyenne et légère modification de variance
    shift = rng.standard_normal(d) * 1.5
    C = rng.standard_normal((d, d)) * 0.1
    Sigma_x1 = Sigma_x0 + C @ C.T

    mu_y0 = rng.standard_normal(d) * 2
    mu_x0 = rng.standard_normal(d) * 2
    mu_x1 = mu_x0 + shift

    X0 = rng.multivariate_normal(mu_x0, Sigma_x0, N)
    Y0 = rng.multivariate_normal(mu_y0, Sigma_y0, N)
    X1 = rng.multivariate_normal(mu_x1, Sigma_x1, N_fut)

    # Corrections
    Y1_cdf = cdf_t_multivar(X0, Y0, X1)
    Y1_ot, _ = transport_optimal(X0, Y0, X1, method='emd', normalize=True)
    Y1_sb, _ = schrodinger_bridge(X0, Y0, X1, reg=reg_sb, normalize=True)

    results = {}
    R_target = np.corrcoef(Y0.T) if d > 1 else None

    for name, Y1 in [('cdf_t', Y1_cdf), ('ot', Y1_ot), ('sb', Y1_sb)]:
        frob_corr = correlation_frobenius(Y0, Y1) if d > 1 else 0.0
        mean_err = float(np.linalg.norm(Y1.mean(axis=0) - mu_y0))

        a_w = np.ones(len(Y1)) / len(Y1)
        b_w = np.ones(len(Y0)) / len(Y0)
        M_w = ot.dist(Y1, Y0, metric='euclidean')
        wass = float(ot.emd2(a_w, b_w, M_w))

        results[name] = {
            'frobenius_corr': frob_corr,
            'wasserstein': wass,
            'mean_error': mean_err,
        }

    meta = {
        'X0': X0, 'Y0': Y0, 'X1': X1,
        'mu_y0': mu_y0, 'Sigma_y0': Sigma_y0,
        'mu_x0': mu_x0, 'Sigma_x0': Sigma_x0,
        'mu_x1': mu_x1, 'shift': shift,
    }
    return results, meta