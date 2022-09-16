import numpy as np

def production(A, S, P, ETP):
    """
    A : capacité du réservoir sol 
    S : niveau du réservoir
    P : pluie
    ETP : ETP
    Pn : pluie nette
    ETR : ETR
    """
    Satur = S / A
    if P >= ETP:
        P1 = (P - ETP)
        Pn = P1 * Satur**2
        ETR = 0
        S = S + A * np.tanh(P1 / A) * ((1 - (S / A)**2)/ (1 + (S / A) * np.tanh(P / A)))
    else:
        E1 = (ETP - P)
        Pn = 0
        ETR = E1 * Satur * (2 - Satur)
        S = S - E1 *( S / (E1 + A /(2 - S / A)))
    return Pn, ETR, S

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
    C = H0 / (H0 + R)
    TGHM = 1 / (1 - np.exp(- np.log(2) / (N * THG)))
    H = H0 * (TGHM - 1) / (TGHM + H0 / R)
    Qi = R * np.log(1 + H / (TGHM * R))
    Qr = (H0 - H)  - Qi
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
    Qg1 = np.minimum(G0 * (1 - np.exp(-np.log(2) / (N* TG1))), G0)
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