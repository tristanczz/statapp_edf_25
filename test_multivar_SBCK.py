import numpy as np
from SBCK import CDFT as OT

def correction_OT_multivariee(mod_hist, obs_hist, mod_future):
    """
    mod_hist, obs_hist, mod_future : arrays de shape (n_samples, n_variables)
    Exemple : température + vent
    """
    ot = OT()
    futur_corrige = ot.fit_transform(mod_hist, obs_hist, mod_future)
    return futur_corrige