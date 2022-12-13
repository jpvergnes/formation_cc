from module.utils import *
import os
import pandas as pd
import numpy as np
import ipywidgets as widgets
from ipywidgets.widgets import HBox, VBox, Label
from scipy.optimize import minimize
from scipy.stats import linregress
import matplotlib.pyplot as plt
import pymannkendall as mk
from IPython.display import display
import matplotlib

class App():
    def __init__(self, flow, level):
        self.area = 524
        self.start = 365*4
        bounds = [
            [1, 650],
            [6, 1000],
            [0.15, 25],
            [6, 70]
        ]
        self.bounds = bounds
        self.flow = flow.squeeze().values
        self.level = level.squeeze().values
        self.sliderA = widgets.FloatSlider(
            value=180,
            min=bounds[0][0],
            max=bounds[0][1],
            continuous_update=False
        )

        self.widgetA = VBox(
            [
                Label('Progressive storage capacity (mm)'),
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
                Label('Runoff/percolation repartition level (mm)'),
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
                Label('Half-percolation time (month)'),
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
                Label('Half-recession time (month)'),
                self.sliderTG1
            ]
        )
        self.baseNiv = widgets.FloatText(120)
        self.widgetbaseNiv = VBox(
            [
                Label('Base groundwater level (m)'),
                self.baseNiv
            ]
        )
        self.coeffEmmag = widgets.FloatText(0.3)
        self.widgetcoeffEmmag = VBox(
            [
                Label("Specific Yield (-)"),
                self.coeffEmmag
            ]
        )
        self.set_observe()
        
        self.textKGE = widgets.FloatText(0, description='KGE - river flow')
        self.textNash = widgets.FloatText(0, description='Nash - river flow')
        self.textNashNiv = widgets.FloatText(0, description='Nash - groundwater level')
        self.btnOpti = widgets.Button(
            description='Optimize',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Optimize',
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
        level = self.level[self.start:]
        flow = self.flow[self.start:]
        
        sim_h = h[~np.isnan(level)]
        obs_h = level[~np.isnan(level)]
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

        sim_q = q[~np.isnan(flow)]
        obs_q = flow[~np.isnan(flow)]
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
        level = self.level[self.start:]
        flow = self.flow[self.start:]
        if value == -9999.:
            self.result = pd.DataFrame(
                {
                    'Flow':q,
                    'level':h
                },
                index = self.dates[self.start:]
            )
        else:
            deb_os = pd.DataFrame(
                {
                    'obs':flow,
                    'sim':q,
                },
                index=self.dates[self.start:]
            )
            deb_os = deb_os.dropna()
            if self.xlims is not None:
                xlims = self.ax1.get_xlim()
            self.ax1.cla()
            deb_os.plot(ax=self.ax1)
            self.ax1.set_title("River flow - Selle at Plachy")
            self.ax1.set_ylabel("m3/s")
            self.ax1.grid()
            self.textNash.value = nash(deb_os['sim'], deb_os['obs']).round(2)
            self.textKGE.value = kge(deb_os['sim'], deb_os['obs']).round(2)

            deb_os = pd.DataFrame(
                {
                    'obs':level,
                    'sim':h,
                },
                index=self.dates[self.start:]
            )
            deb_os = deb_os.dropna()
            self.ax2.cla()
            deb_os.plot(ax=self.ax2)
            self.ax2.set_title("Groundwater level - Morvillers monitoring well")
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
                R, THG, 30.5,#self.daysinmonth[istep],
                Pn, H
            )
            Qg1, G = souterrain(
                TG1, 30.5,#self.daysinmonth[istep],
                Qi, G
            )
            Qrl.append(Qr)
            Qg1l.append(Qg1)
            h.append(G)
        q = np.array(Qg1l) + np.array(Qrl)
        q = q * 1e-3 / 86400 * self.area * 1e6
        h = np.array(h)
        return q, h


class PlotSim():
    def __init__(self):
        models = ['{0} / {1}'.format(model[0], model[1]) for model in drias2020]
        models = list(set(models))
        models.sort()
        self.selectModels = widgets.SelectMultiple(
            options=models,
            description='Models :',
            disabled=False,
            rows=len(models)
        )
        self.selectRcps = widgets.SelectMultiple(
            options=['rcp2.6', 'rcp4.5', 'rcp8.5'],

            description='RCPs :',
            disabled=False
        )
        self.selectParameter = widgets.Dropdown(
            options=['River flow', 'Groundwater level'],
            description='Parameter :',
            disabled=False
        )
        self.selectIndic = widgets.Dropdown(
            options=[
                'Daily',
                'Monthly mean',
                'Monthly minimum',
                'Monthly maximum',
                'Annual mean',
                'Annual minimum',
                'Annual maximum'
            ],
            description='Indicateur :',
            disabled=False
        )
        self.selectPlot = widgets.Dropdown(
            options=['Spaghetti', 'Q5/Median/Q95', 'Min/Mean/Max'],
            description='Tracé :',
            disabled=False
        )

        self.checkAnom = widgets.Checkbox(
            value=False,
            description="Anomaly"
        )
        self.btnPlot = widgets.Button(
            description='Tracer',
            disabled=False,
            icon='check'
        )

        self.btnPlot.on_click(self.main_sim)
        self.col_para = widgets.VBox(
            [
                self.selectModels,
                self.selectRcps,
                self.selectParameter,
                self.selectIndic,
                self.selectPlot,
                self.checkAnom,
                self.btnPlot,
            ]
        )
        plt.ioff()
        self.fig, self.ax = plt.subplots(1, figsize=(10,4))
        self.fig.canvas.header_visible = False
        plt.ion()
        self.main = widgets.HBox([self.col_para, self.fig.canvas])
        self.label = widgets.Label("Mann-Kendall test (only for annual indicators) : ")
        self.out = widgets.Output(layout=widgets.Layout(overflow='visible', width="1500"))

    def main_sim(self, b):
        self.btnPlot.disabled = True
        select = self.check_sim()
        df = {}
        for sel, f in select.items():
            df[tuple(sel.split('_'))] = pd.read_csv(f, parse_dates=True, index_col=0)
        if df:
            df = pd.concat(df, axis=1, names=['Model', 'RCP', 'Parameter'])
            df = df.xs(key=self.selectParameter.value, level='Parameter', axis=1)

            df = self.transform_indic(df)
            df = self.transform_anomaly(df)
            df = self.transform_ensemble(df)
            self.plot_sim(df)
            self.test_mk(df)
        self.btnPlot.disabled = False

    def transform_indic(self, df):
        indic = self.selectIndic.value
        if indic == 'Monthly mean':
            df = df.resample('M').mean()
        elif indic == 'Monthly minimum':
            df = df.resample('M').min()
        elif indic == 'Monthly maximum':
            df = df.resample('M').max()
        elif indic == 'Annual mean':
            df = df.resample('Y').mean()
        elif indic == 'Annual minimum':
            df = df.resample('Y').min()
        elif indic == 'Annual maximum':
            df = df.resample('Y').max()
        return df
    
    def transform_anomaly(self, df):
        anom = self.checkAnom.value
        if anom:
            hist = df.loc['1976':'2005'].mean()
            df = df - hist
            return df.loc['2005':]
        return df
    
    def transform_ensemble(self, df, axis=1):
        ensemble = self.selectPlot.value
        df2 = {}
        if ensemble == "Q5/Median/Q95":
            df2['q5'] = df.groupby(level="RCP", axis=axis).quantile(0.05)
            df2['Median'] = df.groupby(level="RCP", axis=axis).quantile(0.5)
            df2['q95'] = df.groupby(level="RCP", axis=axis).quantile(0.95)
            df = pd.concat(df2, axis=1)
        elif ensemble == 'Min/Mean/Max':
            df2['Min'] = df.groupby(level="RCP", axis=axis).min()
            df2['Mean'] = df.groupby(level="RCP", axis=axis).mean()
            df2['Max'] = df.groupby(level="RCP", axis=axis).max()
            df = pd.concat(df2, axis=1)
        return df
            
    def plot_sim(self, df):
        self.ax.clear()
        if self.selectParameter.value == "River flow":
            self.ax.set_ylabel("m3/s")
        elif self.selectParameter.value == "Groundwater level":
            self.ax.set_ylabel("m")
        ensemble = self.selectPlot.value
        if ensemble == "Q5/Median/Q95":
            for rcp in np.unique(df.columns.get_level_values('RCP')):
                df.loc[:, ("Median", rcp)].plot(ax=self.ax, legend=False, grid=True)
                self.ax.fill_between(
                    df.index,
                    df.loc[:, ("q5", rcp)],
                    df.loc[:, ("q95", rcp)],
                    alpha=0.5
            )
        elif ensemble == "Min/Mean/Max":
            for rcp in np.unique(df.columns.get_level_values('RCP')):
                df.loc[:, ("Mean", rcp)].plot(ax=self.ax, legend=False, grid=True)
                self.ax.fill_between(
                    df.index,
                    df.loc[:, ("Min", rcp)],
                    df.loc[:, ("Max", rcp)],
                    alpha=0.5
            )
        else:
            df.plot(ax=self.ax, legend=False, grid=True)
        self.ax.legend(loc="upper left", bbox_to_anchor=(1.02, 0, 0.07, 1), prop={'size':6})
        self.fig.tight_layout()
    
    def check_sim(self):
        models = self.selectModels.value
        rcps = self.selectRcps.value
        select = {}
        for model in models:
            gcm, rcm = model.split(' / ')
            for rcp in rcps:
                f = 'results_cc/{0}_{1}_{2}.csv'.format(gcm, rcm, rcp)
                if os.path.exists(f):
                    select['{0} / {1}_{2}'.format(gcm, rcm, rcp)] = f
        return select
    
    def test_mk(self, df):
        indic = self.selectIndic.value
        dfout = {}
        if "annuel" in indic:
            for column in df.columns:
                serie = {}
                result = mk.original_test(df.loc[:, column].dropna().values)
                serie['trend'] = result.trend
                serie['p-value'] = result.p.round(4)
                serie['slope'] = result.slope.round(4)
                serie = pd.Series(serie)
                dfout[column] = serie
            dfout = pd.concat(dfout, axis=1)
            with self.out:
                self.out.clear_output()
                display(dfout)

class PlotSim2():
    def __init__(self):
        models = ['{0} / {1}'.format(model[0], model[1]) for model in drias2020]
        models = list(set(models))
        models.sort()
        self.selectModels = widgets.SelectMultiple(
            options=models,
            description='Models :',
            disabled=False,
            rows=len(models)
        )
        self.selectRcps = widgets.SelectMultiple(
            options=['rcp2.6', 'rcp4.5', 'rcp8.5'],
    
            description='RCPs :',
            disabled=False
        )
        self.selectParameter = widgets.Dropdown(
            options=['River flow', 'Groundwater level'],
            description='Parameter :',
            disabled=False
        )
        self.selectIndic = widgets.Dropdown(
            options=[
                'Daily',
                'Monthly mean',
                'Monthly minimum',
                'Monthly maximum',
            ],
            description='Indicateur :',
            disabled=False
        )
        self.selectPeriod = widgets.SelectMultiple(
            options=[
                '2021-2050',
                '2041-2070',
                '2071-2100',
            ],
            description='Période :',
            disabled=False
        )
        self.selectPlot = widgets.Dropdown(
            options=['Spaghetti', 'Q5/Median/Q95', 'Min/Mean/Max'],
            description='Tracé :',
            disabled=False
        )
    
        self.checkAnom = widgets.Checkbox(
            value=False,
            description="Anomaly"
        )
        self.btnPlot = widgets.Button(
            description='Tracer',
            disabled=False,
            icon='check'
        )
    
        self.btnPlot.on_click(self.main_sim)
        self.col_para = widgets.VBox(
            [
                self.selectModels,
                self.selectRcps,
                self.selectParameter,
                self.selectIndic,
                self.selectPlot,
                self.selectPeriod,
                self.checkAnom,
                self.btnPlot,
            ]
        )
        plt.ioff()
        self.fig, self.ax = plt.subplots(1, figsize=(10,4))
        self.fig.canvas.header_visible = False
        plt.ion()
        self.main = widgets.HBox([self.col_para, self.fig.canvas])
        self.label = widgets.Label("30-years interannual mean : ")
        self.out = widgets.Output(layout=widgets.Layout(overflow='visible', width="1500"))

    def main_sim(self, b):
        self.btnPlot.disabled = True
        select = self.check_sim()
        df = {}
        for sel, f in select.items():
            df[tuple(sel.split('_'))] = pd.read_csv(f, parse_dates=True, index_col=0)
        if df:
            df = pd.concat(df, axis=1, names=['Model', 'RCP', 'Parameter'])
            df = df.xs(key=self.selectParameter.value, level='Parameter', axis=1)
            df = self.transform_indic(df)
            df = self.transform_cycle(df)
            if self.checkAnom.value:
                for period in self.selectPeriod.value:
                    df[period] = df[period] - df['1976-2005']
                    self.dfmean.loc[(period, slice(None), slice(None))] = (
                        self.dfmean[period].values - self.dfmean['1976-2005'].values
                    )
                del df['1976-2005']
                del self.dfmean['1976-2005']
            df = self.transform_ensemble(df)
            self.dfmean = self.transform_ensemble(self.dfmean, axis=0)
            self.plot_sim(df)
            self.test_stats()
        self.btnPlot.disabled = False

    def transform_indic(self, df):
        indic = self.selectIndic.value
        if indic == 'Monthly mean':
            df = df.resample('M').mean()
        elif indic == 'Monthly minimum':
            df = df.resample('M').min()
        elif indic == 'Monthly maximum':
            df = df.resample('M').max()
        return df

    def transform_cycle(self, df):
        df2 = {
            '1976-2005': df.loc['1976':'2005'],
            '2021-2050': df.loc['2021':'2050'],
            '2041-2070': df.loc['2041':'2070'],
            '2071-2100': df.loc['2071':'2100']
        }
        indic = self.selectIndic.value
        df3 = {}
        if 'mensuel' in indic:
            dfc = df2['1976-2005']
            df3['1976-2005'] = dfc.groupby(dfc.index.month).mean()
            for period in self.selectPeriod.value:
                dfc = df2[period]
                df3[period] = dfc.groupby(dfc.index.month).mean()
        else:
            dfc = df2['1976-2005']
            df3['1976-2005'] = dfc.groupby(dfc.index.dayofyear).mean()
            for period in self.selectPeriod.value:
                dfc = df2[period]
                df3[period] = dfc.groupby(dfc.index.dayofyear).mean()
        df = pd.concat(df3, axis=1, names=['Period'])
        self.dfmean = df.mean(axis=0)
        return df

    def transform_ensemble(self, df, axis=1):
        ensemble = self.selectPlot.value
        df2 = {}
        if ensemble == "Q5/Median/Q95":
            df2['q5'] = df.groupby(level=("Period", "RCP"), axis=axis).quantile(0.05)
            df2['Median'] = df.groupby(level=("Period", "RCP"), axis=axis).quantile(0.5)
            df2['q95'] = df.groupby(level=("Period", "RCP"), axis=axis).quantile(0.95)
            df = pd.concat(df2, axis=1)
        elif ensemble == 'Min/Mean/Max':
            df2['Min'] = df.groupby(level=("Period", "RCP"), axis=axis).min()
            df2['Mean'] = df.groupby(level=("Period", "RCP"), axis=axis).mean()
            df2['Max'] = df.groupby(level=("Period", "RCP"), axis=axis).max()
            df = pd.concat(df2, axis=1)
        return df

    def plot_sim(self, df):
        self.ax.clear()
        if self.selectParameter.value == "River flow":
            self.ax.set_ylabel("m3/s")
        elif self.selectParameter.value == "Groundwater level":
            self.ax.set_ylabel("m")
        ensemble = self.selectPlot.value
        if ensemble == "Q5/Median/Q95":
            for rcp in np.unique(df.columns.get_level_values('RCP')):
                for period in np.unique(df.columns.get_level_values('Period')):
                    self.ax.plot(df.index, df.loc[:, ("Median", period, rcp)], label=(period, rcp))
                    self.ax.fill_between(
                        df.index,
                        df.loc[:, ("q5", period, rcp)],
                        df.loc[:, ("q95", period, rcp)],
                        alpha=0.5
                    )
        elif ensemble == "Min/Mean/Max":
            for rcp in np.unique(df.columns.get_level_values('RCP')):
                for period in np.unique(df.columns.get_level_values('Period')):
                    self.ax.plot(df.index, df.loc[:, ("Mean", period, rcp)], label=(period, rcp))
                    self.ax.fill_between(
                        df.index,
                        df.loc[:, ("Min", period, rcp)],
                        df.loc[:, ("Max", period, rcp)],
                        alpha=0.5
                    )
        else:
            for column in df.columns:
                df2 = df.iloc[:-1, :]
                self.ax.plot(df2.index, df2.loc[:, column], label=column)
        if self.selectIndic.value == "Daily":
            self.ax.set_xticks([i for i in range(15, 370, 30)])
        else:
            self.ax.set_xticks([i for i in range(1, 13)])
        self.ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
        self.ax.grid()
        self.ax.legend(loc="upper left", bbox_to_anchor=(1.02, 0, 0.07, 1), prop={'size':6})
        self.fig.tight_layout()

    def check_sim(self):
        models = self.selectModels.value
        rcps = self.selectRcps.value
        select = {}
        for model in models:
            gcm, rcm = model.split(' / ')
            for rcp in rcps:
                f = 'results_cc/{0}_{1}_{2}.csv'.format(gcm, rcm, rcp)
                if os.path.exists(f):
                    select['{0} / {1}_{2}'.format(gcm, rcm, rcp)] = f
        return select
    
    def test_stats(self):
        with self.out:
            self.out.clear_output()
            display(self.dfmean)
            
class PlotSim3(PlotSim2):
    def __init__(self):
        super().__init__()
        self.selectParameter = widgets.Dropdown(
            options=['Rainfall', 'PET'],
            description='Parameter :',
            disabled=False
        )
            
        self.selectIndic = widgets.Dropdown(
            options=[
                'Daily',
                'Monthly cumulative',
                'Annual cumulative',
            ],
            description='Indicateur :',
            disabled=False
        )
        self.col_para = widgets.VBox(
            [
                self.selectModels,
                self.selectRcps,
                self.selectParameter,
                self.selectIndic,
                self.selectPlot,
                self.selectPeriod,
                self.checkAnom,
                self.btnPlot,
            ]
        )
        self.main = widgets.HBox([self.col_para, self.fig.canvas])
        
    def transform_indic(self, df):
        indic = self.selectIndic.value
        if indic == 'Monthly cumulative':
            df = df.resample('M').sum()
        elif indic == 'Annual cumulative':
            df = df.resample('Y').sum()
        return df
    
    def transform_cycle(self, df):
        df2 = {
            '1976-2005': df.loc['1976':'2005'],
            '2021-2050': df.loc['2021':'2050'],
            '2041-2070': df.loc['2041':'2070'],
            '2071-2100': df.loc['2071':'2100']
        }
        indic = self.selectIndic.value
        df3 = {}
        if 'mensuel' in indic:
            dfc = df2['1976-2005']
            df3['1976-2005'] = dfc.groupby(dfc.index.month).mean()
            for period in self.selectPeriod.value:
                dfc = df2[period]
                df3[period] = dfc.groupby(dfc.index.month).mean()
        elif "Daily" in indic:
            dfc = df2['1976-2005']
            df3['1976-2005'] = dfc.groupby(dfc.index.dayofyear).mean()
            for period in self.selectPeriod.value:
                dfc = df2[period]
                df3[period] = dfc.groupby(dfc.index.dayofyear).mean()
        else:
            dfc = df2['1976-2005']
            df3['1976-2005'] = dfc
            for period in self.selectPeriod.value:
                dfc = df2[period]
                df3[period] = dfc
        df = pd.concat(df3, axis=1, names=['Period'])
        self.dfmean = df.mean(axis=0)
        return df
    
    def check_sim(self):
        models = self.selectModels.value
        rcps = self.selectRcps.value
        select = {}
        for model in models:
            gcm, rcm = model.split(' / ')
            for rcp in rcps:
                fetp = 'sims_cc/{0}_{1}_{2}/PET_Selle_et_Morvillers_{0}_{1}_{2}'.format(gcm, rcm, rcp)
                fplu = 'sims_cc/{0}_{1}_{2}/Rainfall_Selle_et_Morvillers_{0}_{1}_{2}'.format(gcm, rcm, rcp)
                if os.path.exists(fetp) and os.path.exists(fplu):
                    select['{0} / {1}_{2}'.format(gcm, rcm, rcp)] = (fetp, fplu)
        return select
    
    def main_sim(self, b):
        self.btnPlot.disabled = True
        select = self.check_sim()
        df = {}
        for sel, f in select.items():
            fetp = f[0]
            fplu = f[1]
            df[tuple(sel.split('_') + ['Rainfall'])] = pd.read_csv(
                fplu, parse_dates=True, index_col=0, delim_whitespace=True, dayfirst=True
            )
            df[tuple(sel.split('_') + ['PET'])] = pd.read_csv(
                fetp, parse_dates=True, index_col=0, delim_whitespace=True, dayfirst=True
            )
        if df:
            df = pd.concat(df, axis=1, names=['Model', 'RCP', 'Parameter'])
            df = df.xs(key=self.selectParameter.value, level='Parameter', axis=1)
            df = self.transform_indic(df)
            df = self.transform_cycle(df)
            if self.checkAnom.value:
                for period in self.selectPeriod.value:
                    df[period] = df[period] - df['1976-2005']
                    self.dfmean.loc[(period, slice(None), slice(None))] = (
                        self.dfmean[period].values - self.dfmean['1976-2005'].values
                    )
                del df['1976-2005']
                del self.dfmean['1976-2005']
            df = self.transform_ensemble(df)
            self.dfmean = self.transform_ensemble(self.dfmean, axis=0)
            self.plot_sim(df)
            self.test_stats()
        self.btnPlot.disabled = False

    def plot_sim(self, df):
        self.ax.clear()
        if self.selectParameter.value == "River flow":
            self.ax.set_ylabel("m3/s")
        elif self.selectParameter.value == "Groundwater level":
            self.ax.set_ylabel("m")
        indic = self.selectIndic.value
        if indic == 'Monthly cumulative':
            self.ax.set_ylabel("mm/month")
        elif indic == 'Annual cumulative':
            self.ax.set_ylabel("mm/year")
        else:
            self.ax.set_ylabel("mm/day")
        ensemble = self.selectPlot.value
        if ensemble == "Q5/Median/Q95":
            for rcp in np.unique(df.columns.get_level_values('RCP')):
                for period in np.unique(df.columns.get_level_values('Period')):
                    self.ax.plot(df.index, df.loc[:, ("Median", period, rcp)], label=(period, rcp))
                    self.ax.fill_between(
                        df.index,
                        df.loc[:, ("q5", period, rcp)],
                        df.loc[:, ("q95", period, rcp)],
                        alpha=0.5
                    )
        elif ensemble == "Min/Mean/Max":
            for rcp in np.unique(df.columns.get_level_values('RCP')):
                for period in np.unique(df.columns.get_level_values('Period')):
                    self.ax.plot(df.index, df.loc[:, ("Mean", period, rcp)], label=(period, rcp))
                    self.ax.fill_between(
                        df.index,
                        df.loc[:, ("Min", period, rcp)],
                        df.loc[:, ("Max", period, rcp)],
                        alpha=0.5
                    )
        else:
            for column in df.columns:
                df2 = df.iloc[:-1, :]
                self.ax.plot(df2.index, df2.loc[:, column], label=column)
        if self.selectIndic.value == "Daily":
            self.ax.set_xticks([i for i in range(15, 370, 30)])
        else:
            self.ax.set_xticks([i for i in range(1, 13)])
        self.ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
        self.ax.grid()
        self.ax.legend(loc="upper left", bbox_to_anchor=(1.02, 0, 0.07, 1), prop={'size':6})
        self.fig.tight_layout()
