import numpy as np

drias2020 = [
    ('CNRM-CM5-LR', 'ALADIN63', 'rcp2.6'),
    ('CNRM-CM5-LR', 'ALADIN63', 'rcp4.5'),
    ('CNRM-CM5-LR', 'ALADIN63', 'rcp8.5'),
    ('CNRM-CM5-LR', 'RACMO22E', 'rcp2.6'),
    ('CNRM-CM5-LR', 'RACMO22E', 'rcp4.5'),
    ('CNRM-CM5-LR', 'RACMO22E', 'rcp8.5'),
    ('EC-EARTH', 'RACMO22E', 'rcp2.6'),
    ('EC-EARTH', 'RACMO22E', 'rcp4.5'),
    ('EC-EARTH', 'RACMO22E', 'rcp8.5'),
    ('EC-EARTH', 'RCA4', 'rcp2.6'),
    ('EC-EARTH', 'RCA4', 'rcp4.5'),
    ('EC-EARTH', 'RCA4', 'rcp8.5'),
    ('HadGEM2-ES', 'CCLM4-8-17', 'rcp4.5'),
    ('HadGEM2-ES', 'CCLM4-8-17', 'rcp8.5'),
    ('HadGEM2-ES', 'RegCM4-6', 'rcp2.6'),
    ('HadGEM2-ES', 'RegCM4-6', 'rcp8.5'),
    ('IPSL-CM5A-MR', 'RCA4', 'rcp4.5'),
    ('IPSL-CM5A-MR', 'RCA4', 'rcp8.5'),
    ('IPSL-CM5A-MR', 'WRF381P', 'rcp4.5'),
    ('IPSL-CM5A-MR', 'WRF381P', 'rcp8.5'),
    ('MPI-ESM-LR', 'CCLM4-8-17', 'rcp2.6'),
    ('MPI-ESM-LR', 'CCLM4-8-17', 'rcp4.5'),
    ('MPI-ESM-LR', 'CCLM4-8-17', 'rcp8.5'),
    ('MPI-ESM-LR', 'REMO2009', 'rcp2.6'),
    ('MPI-ESM-LR', 'REMO2009', 'rcp4.5'),
    ('MPI-ESM-LR', 'REMO2009', 'rcp8.5'),
    ('NorESM1-M', 'HIRHAM5', 'rcp4.5'),
    ('NorESM1-M', 'HIRHAM5', 'rcp8.5'),
    ('NorESM1-M', 'REMO2015', 'rcp2.6'),
    ('NorESM1-M', 'REMO2015', 'rcp8.5')
]



def production(A, S, P, ETP):
    """
    A : capacité du réservoir sol 
    S : niveau du réservoir
    P : pluie
    ETP : ETP
    Pn : pluie nette
    ETR : ETR
    """
    Peff = 0
    ETR1 = 0
    Pn = P
    Snew = S
    if ETP != 0:
        Pn = P - ETP
        ETR1 = min(P- Pn, P)
    if Pn > 0:
        Peff = Pn
    ETP = 0
    if Pn < 1e-5:
        ETP = - Pn
    if abs(Peff - ETP) < 1e-6:
        ETR2 = ETP
    elif ETP > Peff:
        E1 = (ETP - Peff)
        Pn = 0
        Snew = S * (1 - np.tanh(E1 / A)) / (1 + (1 - S / A) * np.tanh(E1 / A))
        Snew = np.maximum(0, Snew)
        ETR2 = P + S - Snew
    elif Peff > ETP:
        P1 = (Peff - ETP)
        ETR2 = 0
        Snew = (S + A * np.tanh(P1 / A)) / (1 + S / A * np.tanh(P1 / A))
        DEBORD = np.maximum(Snew - A, 0)
        Pn = P1 - (Snew - S) + DEBORD
    return Pn, ETR1 + ETR2, Snew

def intermediaire(R, THG, N, Pn, HP):
    """
    R : hauteur ruissellement/percolation
    THG : temps de demi-vie percolation vers la nappe
    N : nombre de pas de temps dans le mois
    Pn : pluie nette
    HP : hauteur réservoir intermédiaire en début de pas de temps
    H : hauteur réservoir intermédiaire
    Qr : écoulement rapide
    Qi : écoulement lent
    """
    H0 = HP + Pn
    TGHM = 1 / (1 - np.exp(- np.log(2) / (N * THG)))
    TGHM = max(TGHM, 1)
    D = 1 - 1 / TGHM
    Qi = R * np.log(1 + H0 / (TGHM * R))
    H = H0 * D / (1 + H0 / (TGHM * R))
    Qr = H0 - H - Qi
    Qr = max(Qr, 0)
    Qi = H0 - H - Qr
    return Qr, Qi, H

def souterrain(TG1, N, Qi, GP):
    """
    TG1 : temps de demi-tarissement vers la nappe
    N : nombre de pas de temps dans le mois
    Qi : écoulement lent
    GP : hauteur d'eau souterraine en début de pas de temps
    Qg1 : écoulement lent du réservoir
    G : hauteur d'eau souterraine
    """
    G0 = GP + Qi
    Qg1 = np.minimum(G0 * (1 - np.exp(-np.log(2) / (N * TG1))), G0)
    Qg1 = np.maximum(Qg1, 0)
    G = np.maximum(G0 - Qg1, 0)
    return Qg1, G

def kge(sim, obs):
    r = np.corrcoef(sim, obs)[0, 1]
    a = sim.std() / obs.std()
    b = sim.mean() / obs.mean()
    Ed = np.sqrt((r - 1)**2 + (a - 1)**2 + (b - 1)**2)
    return 1 - Ed

def nash(sim, obs):
    return (
        1 - 
        np.sum((sim - obs)**2) / 
        np.sum((obs - obs.mean())**2)
    )