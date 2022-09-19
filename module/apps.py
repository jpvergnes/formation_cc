from module.utils import *
import pandas as pd
import numpy as np
import ipywidgets as widgets
from ipywidgets.widgets import HBox, VBox, Label
from scipy.optimize import minimize
from scipy.stats import linregress
import matplotlib.pyplot as plt

class App():
    def __init__(self, debit, niveau):
        self.area = 524
        self.start = 365*4
        bounds = [
            [1, 650],
            [6, 1000],
            [0.15, 25],
            [6, 70]
        ]
        self.bounds = bounds
        self.debit = debit.squeeze().values
        self.niveau = niveau.squeeze().values
        self.sliderA = widgets.FloatSlider(
            value=180,
            min=bounds[0][0],
            max=bounds[0][1],
            continuous_update=False
        )

        self.widgetA = VBox(
            [
                Label('Capacité du réservoir sol progressif (mm)'),
                self.sliderA
            ]
        )
        self.sliderR = widgets.FloatSlider(
            min=bounds[1][0],
            max=bounds[1][1],
            value=600,
            continuous_update=False,
        )
        self.widgetR = VBox(
            [
                Label('Hauteur de ruissellement/percolation (mm)'),
                self.sliderR
            ]
        )
        self.sliderTHG = widgets.FloatSlider(
            min=bounds[2][0],
            max=bounds[2][1],
            value=5,
            continuous_update=False
        )
        self.widgetTHG = VBox(
            [
                Label('Temps de demi-percolation du réservoir intermédiaire (mois)'),
                self.sliderTHG
            ]
        )
        self.sliderTG1 = widgets.FloatSlider(
            min=bounds[3][0],
            max=bounds[3][1],
            value=10,
            continuous_update=False
        )
        self.widgetTG1 = VBox(
            [
                Label('Temps de demi-tarissement du réservoir souterrain (mois)'),
                self.sliderTG1
            ]
        )
        self.baseNiv = widgets.FloatText(120)
        self.widgetbaseNiv = VBox(
            [
                Label('Niveau de base (m)'),
                self.baseNiv
            ]
        )
        self.coeffEmmag = widgets.FloatText(0.3)
        self.widgetcoeffEmmag = VBox(
            [
                Label("Coefficient d'emmagasinement (-)"),
                self.coeffEmmag
            ]
        )
        self.set_observe()
        
        self.textKGE = widgets.FloatText(0, description='KGE Débit')
        self.textNash = widgets.FloatText(0, description='Nash Débit')
        self.textKGE = widgets.FloatText(0, description='KGE Débit')
        self.textNashNiv = widgets.FloatText(0, description='Nash Niveau')
        self.btnOpti = widgets.Button(
            description='Optimiser',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Optimiser',
            icon='check'
        )
        self.btnOpti.on_click(self.optimize)
        self.out = widgets.Output(layout={'border': '1px solid black'})
        self.xlims = None

        plt.ioff()
        self.fig, axes = plt.subplots(2)
        self.ax1 = axes[0]
        self.ax2 = axes[1]
        self.ax2.sharex(self.ax1)
        self.fig.canvas.header_visible = False
        plt.ion()
    
    def set_meteo(self, pluie, etp):
        self.pluie = pluie.squeeze().values
        self.etp = etp.squeeze().values
        self.pnm = self.pluie - self.etp
        self.dates = pluie.index
        self.daysinmonth = [date.daysinmonth for date in self.dates]
        
    def set_observe(self):
        self.sliderA.observe(self.run, names='value')
        self.baseNiv.observe(self.run, names='value')
        self.sliderTG1.observe(self.run, names='value')
        self.coeffEmmag.observe(self.run, names='value')
        self.sliderR.observe(self.run, names='value')
        self.sliderTHG.observe(self.run, names='value')
    
    def set_unobserve(self):
        self.sliderA.unobserve_all()
        self.baseNiv.unobserve_all()
        self.sliderTG1.unobserve_all()
        self.coeffEmmag.unobserve_all()
        self.sliderR.unobserve_all()
        self.sliderTHG.unobserve_all()
        
    def optimize(self, b):
        self.btnOpti.disabled = True
        self.sliderA.disabled = True
        self.sliderR.disabled = True
        self.sliderTHG.disabled = True
        self.sliderTG1.disabled = True
        self.baseNiv.disabled = True
        self.coeffEmmag.disabled = True
        self.set_unobserve()
        A = self.sliderA.get_interact_value()
        R = self.sliderR.get_interact_value()
        THG = self.sliderTHG.get_interact_value()
        TG1 = self.sliderTG1.get_interact_value()
        with self.out:
            self.out.clear_output()
            res = minimize(
                self.func,
                [A, R, THG, TG1],
                bounds=self.bounds,
                method='L-BFGS-B',
                #method='Nelder-Mead',
                tol=1e-6,
                options={
                    'maxiter':250,
                }
            )
        self.btnOpti.disabled = False
        self.sliderA.disabled = False
        self.sliderR.disabled = False
        self.sliderTHG.disabled = False
        self.sliderTG1.disabled = False
        self.baseNiv.disabled = False
        self.coeffEmmag.disabled = False
        self.sliderA.value = res.x[0]
        self.sliderR.value = res.x[1]
        self.sliderTHG.value = res.x[2]
        self.sliderTG1.value = res.x[3]
        self.baseNiv.value = self.niv.round(2)
        self.coeffEmmag.value = 1 /  (self.emm*1000)
        self.set_observe()
        self.run(0)

    def func(self, params, *args):
        A = params[0]
        R = params[1]
        THG = params[2]
        TG1 = params[3]
        qq, hh = self.run_model(A, R, THG, TG1)
        
        q = qq[self.start:]
        h = hh[self.start:]
        niveau = self.niveau[self.start:]
        debit = self.debit[self.start:]
        
        sim_h = h[~np.isnan(niveau)]
        obs_h = niveau[~np.isnan(niveau)]
        df = pd.DataFrame(
            {
                'obs':obs_h,
                'sim':sim_h,
            }
        )
        res = np.polyfit(sim_h, obs_h, 1)
        self.emm = res[0]
        self.niv = res[1]

        sim_h = self.niv + self.emm * sim_h
        nse_h = nash(sim_h, obs_h)
        if nse_h >= -1:
            f_h = np.sqrt(nse_h)
        else:
            f_h = -np.sqrt(-nse_h)

        sim_q = q[~np.isnan(debit)]
        obs_q = debit[~np.isnan(debit)]
        nse_q = nash(sim_q, obs_q)
        if nse_q >= -1:
            f_q = np.sqrt(nse_q)
        else:
            f_q = -np.sqrt(-nse_q)

        f = 5 * f_q + 2 * f_h
        return -f
        
    def run(self, value):
        A = self.sliderA.get_interact_value()
        R = self.sliderR.get_interact_value()
        THG = self.sliderTHG.get_interact_value()
        TG1 = self.sliderTG1.get_interact_value()
        NIV = self.baseNiv.get_interact_value()
        EMM = self.coeffEmmag.get_interact_value()
        q, h = self.run_model(A, R, THG, TG1)
        h = NIV + 1 /(EMM * 1000) * np.array(h)
        q = q[self.start:]
        h = h[self.start:]
        niveau = self.niveau[self.start:]
        debit = self.debit[self.start:]
        if value == -9999.:
            self.result = pd.DataFrame(
                {
                    'Débit':q,
                    'Niveau':h
                },
                index = self.dates[self.start:]
            )
        else:
            deb_os = pd.DataFrame(
                {
                    'obs':debit,
                    'sim':q,
                },
                index=self.dates[self.start:]
            )
            deb_os = deb_os.dropna()
            if self.xlims is not None:
                xlims = self.ax1.get_xlim()
            self.ax1.cla()
            deb_os.plot(ax=self.ax1)
            self.ax1.set_title("Débit - Selle à Plachy")
            self.ax1.set_ylabel("m3/s")
            self.ax1.grid()
            self.textNash.value = nash(deb_os['sim'], deb_os['obs']).round(2)
            self.textKGE.value = kge(deb_os['sim'], deb_os['obs']).round(2)

            deb_os = pd.DataFrame(
                {
                    'obs':niveau,
                    'sim':h,
                },
                index=self.dates[self.start:]
            )
            deb_os = deb_os.dropna()
            self.ax2.cla()
            deb_os.plot(ax=self.ax2)
            self.ax2.set_title("Piézomètre à Morvillers")
            self.ax2.set_ylabel("m")
            self.ax2.grid()
            self.textNashNiv.value = nash(deb_os['sim'], deb_os['obs']).round(2)

            if self.xlims is None:
                self.xlims = self.ax1.get_xlim()
            else:
                self.ax1.set_xlim(xlims[0], xlims[1])
            self.fig.tight_layout()
            self.fig.canvas.draw()

    def run_model(self, A, R, THG, TG1):
        S =  508.73
        G = 419.02
        H = 71.811
        Qrl, Qg1l, h = [], [], []
        for istep in range(len(self.pluie)):
            Pn, ETR, S = production(
                A, S, 
                self.pluie[istep],
                self.etp[istep],
            )
            Qr, Qi, H = intermediaire(
                R, THG, self.daysinmonth[istep],
                Pn, H
            )
            Qg1, G = souterrain(
                TG1, self.daysinmonth[istep],
                Qi, G
            )
            Qrl.append(Qr)
            Qg1l.append(Qg1)
            h.append(G)
        q = np.array(Qg1l) + np.array(Qrl)
        q = q * 1e-3 / 86400 * self.area * 1e6
        h = np.array(h)
        return q, h

#class App2():
#    def __init__(self):
#    widgets.Checkbox(
#        value=False,
#        description='Check me',
#        disabled=False,
#        indent=False
#    )

