import scipy.integrate as integrate
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm as normal_dist
import pygeostat as gs
from sklearn.neighbors import NearestNeighbors
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go

import plotly.io as pio

import os
import warnings
warnings.filterwarnings("ignore")


def lambda34_alpha34_visual(l3_mesh, l4_mesh, skew_surf, kurtosis_surf):
    '''
    Plots skewness and kurtosis vs the shape parameters of lambda distribution (i.e. lambda3 and lambda4)
    '''
    init_notebook_mode()
    scene = dict(
        xaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        yaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        zaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        camera=dict(eye=dict(x=1.8, y=1.8, z=1.8))
    )

    layout = go.Layout(margin=dict(l=0,
                                   r=0,
                                   b=0,
                                   t=0
                                   ), showlegend=False, title=go.layout.Title(
        text='Summary Visualization',
        font=dict(
            family='Courier New, monospace',
            size=20,
            color='#7f7f7f'
        )
    )
    )

    opacity = 1
    axis_font_size = 16
    axis_font_color = '#000000'

    skewness = go.Surface(x=l3_mesh, y=l4_mesh, z=skew_surf, colorscale='Viridis', showscale=True,
                          colorbar=dict(len=0.15, thickness=10, x=0.45, y=0.55),

                          scene='scene1')

    kurtosis = go.Surface(x=l3_mesh, y=l4_mesh, z=kurtosis_surf, colorscale='Viridis', showscale=True,
                          colorbar=dict(len=0.15, thickness=10, x=0.93, y=0.55),
                          scene='scene2')

    locationmap_lambda3 = go.Scatter(x=skew_surf.flatten(),
                                     y=kurtosis_surf.flatten(),
                                     mode='markers',
                                     marker=dict(size=3, color=l3_mesh.flatten(), colorscale='Jet', showscale=True, colorbar=dict(len=0.15, thickness=10, x=0.4, y=0.15, title='lambda3')
                                                 )
                                     )

    locationmap_lambda4 = go.Scatter(x=skew_surf.flatten(),
                                     y=kurtosis_surf.flatten(),
                                     mode='markers',
                                     marker=dict(size=3, color=l4_mesh.flatten(), colorscale='Jet', showscale=True, colorbar=dict(len=0.15, thickness=10, x=0.9, y=0.15, title='lambda4')
                                                 ), xaxis='x2', yaxis='y2'
                                     )

    fig = go.Figure(data=[skewness, kurtosis, locationmap_lambda3,
                          locationmap_lambda4], layout=layout)

    fig['layout'].update(title=' ',
                         height=800, width=1000)

    layout = layout
    fig['layout']['scene1'] = scene
    fig['layout']['scene2'] = scene
    fig['layout']['scene1']['domain'].update({"x": [0.0, 0.5], "y": [0.3, 1]})
    fig['layout']['scene2']['domain'].update({"x": [0.5, 1], "y": [0.3, 1]})
    fig['layout']['scene1']['xaxis'].update({"title": dict(text='lambda_3', font=dict(
        family='Courier New, monospace', size=axis_font_size, color=axis_font_color))})
    fig['layout']['scene1']['yaxis'].update({"title": dict(text='lambda_4', font=dict(
        family='Courier New, monospace', size=axis_font_size, color=axis_font_color))})
    fig['layout']['scene1']['zaxis'].update({"title": dict(text='Skew', font=dict(
        family='Courier New, monospace', size=axis_font_size, color=axis_font_color))})
    fig['layout']['scene2']['xaxis'].update({"title": dict(text='lambda_3', font=dict(
        family='Courier New, monospace', size=axis_font_size, color=axis_font_color))})
    fig['layout']['scene2']['yaxis'].update({"title": dict(text='lambda_4', font=dict(
        family='Courier New, monospace', size=axis_font_size, color=axis_font_color))})
    fig['layout']['scene2']['zaxis'].update({"title": dict(text='Kurtosis', font=dict(
        family='Courier New, monospace', size=axis_font_size, color=axis_font_color))})

    fig['layout']['xaxis'].update({
        "anchor": "y",
        "title": 'Skew',
        "domain": [0.1, 0.4]
    })
    fig['layout']['yaxis'].update({
        "anchor": "x",
        "domain": [0.05, 0.3],
        "title": dict(text='Kurtosis', font=dict(family='Courier New, monospace', size=axis_font_size, color=axis_font_color))
    })

    fig['layout']['xaxis2'] = go.layout.XAxis()
    fig['layout']['xaxis2'].update({
        "anchor": "y2",
        "title": 'Skew',
        "domain": [0.6, 0.9]
    })
    fig['layout']['yaxis2'] = go.layout.YAxis()
    fig['layout']['yaxis2'].update({
        "anchor": "x2",
        "domain": [0.05, 0.3],
        "title": dict(text='Kurtosis', font=dict(family='Courier New, monospace', size=axis_font_size, color=axis_font_color))
    })

    config = {
        'showLink': False,
        'doubleClick': 'reset+autosize',
        'responsive': True,
        'autosizable': True,
        'displayModeBar': True,
        'scrollZoom': False,
        'displayModeBar': True,
        'editable': False}
    _ = iplot(fig, filename='multiple_plots', config=config)


# def beta_function(a,b):
#     '''
#     The function to calculate beta integral required for Lambda distribution calculations
#     '''
#     def main_function(x,a,b):
#         return (x**(a-1)) * ((1-x)**(b-1))

#     result = integrate.quad(main_function, 0, 1, args=(a,b))

#     return result[0]

# def beta_function(a,b):
#     '''
#     The function to calculate beta integral required for Lambda distribution calculations
#     '''
#     import mpmath as mp
#     mp.mp.dps = 50;

#     def main_function(x,a=a,b=b):
#         return (x**(a-1)) * ((1-x)**(b-1))

#     result = mp.quad(main_function, [0, 1])

#     return np.float64(result)

def beta_function(a, b):
    '''
    The function to calculate beta integral required for Lambda distribution calculations
    '''
    import scipy

    return scipy.special.beta(a, b)


class GeneralizedLambdaDist(object):
    '''
    This class provides generalized lambda distribution
    '''

    def __init__(self, lambda1, lambda2, lambda3, lambda4):

        self.l1 = np.float64(lambda1)
        self.l2 = np.float64(lambda2)
        self.l3 = np.float64(lambda3)
        self.l4 = np.float64(lambda4)

    def __str__(self):

        return 'Generalized lambda distribution with 4 parameters'

    # Context manager protocol (i.e. disposable object)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("in __exit__")

    def quantile_function(self, u):
        '''
        Quantile or inverse CDF function

        '''

        if self.l3 == 0.0:
            v1 = np.log(u)
        else:
            v1 = (u**self.l3 - 1) / self.l3

        if self.l4 == 0.0:
            v2 = np.log(1 - u)
        else:
            v2 = ((1 - u)**self.l4 - 1) / self.l4

        s = v1 - v2

        return self.l1 + ((1 / self.l2) * (s))

    def get_moments(self, return_6th=False):

        v1_1 = self.l3 * (self.l3 + 1)
        v1_2 = self.l4 * (self.l4 + 1)
        v1 = (1 / v1_1) - (1 / v1_2)

        v2_1 = 1 / ((self.l3**2) * (2 * self.l3 + 1))
        v2_2 = 1 / ((self.l4**2) * (2 * self.l4 + 1))
        v2_3 = 2 * beta_function(self.l3 + 1, self.l4 + 1) / (self.l3 * self.l4)
        v2 = v2_1 + v2_2 - v2_3

        v3_1 = 1 / ((self.l3**3) * (3 * self.l3 + 1))
        v3_2 = 1 / ((self.l4**3) * (3 * self.l4 + 1))
        v3_3 = 3 * beta_function(2 * self.l3 + 1, self.l4 + 1) / (self.l4 * self.l3**2)
        v3_4 = 3 * beta_function(self.l3 + 1, 2 * self.l4 + 1) / (self.l3 * self.l4**2)
        v3 = v3_1 - v3_2 - v3_3 + v3_4

        v4_1 = 1 / ((self.l3**4) * (4 * self.l3 + 1))
        v4_2 = 1 / ((self.l4**4) * (4 * self.l4 + 1))
        v4_3 = 6 * beta_function(2 * self.l3 + 1, 2 * self.l4 +
                                 1) / (self.l3**2 * self.l4**2)
        v4_4 = 4 * beta_function(3 * self.l3 + 1, self.l4 + 1) / (self.l3**3 * self.l4)
        v4_5 = 4 * beta_function(self.l3 + 1, 3 * self.l4 + 1) / (self.l3 * self.l4**3)
        v4 = v4_1 + v4_2 + v4_3 - v4_4 - v4_5

        if return_6th:
            v5_1 = 1 / ((self.l3**5) * (5 * self.l3 + 1))
            v5_2 = 1 / ((self.l4**5) * (5 * self.l4 + 1))
            v5_3 = 5 * beta_function(4 * self.l3 + 1, self.l4 +
                                     1) / (self.l3**4 * self.l4)
            v5_4 = 10 * beta_function(3 * self.l3 + 1, 2 *
                                      self.l4 + 1) / (self.l3**3 * self.l4**2)
            v5_5 = 10 * beta_function(2 * self.l3 + 1, 3 *
                                      self.l4 + 1) / (self.l3**2 * self.l4**3)
            v5_6 = 5 * beta_function(self.l3 + 1, 4 * self.l4 +
                                     1) / (self.l3 * self.l4**4)
            v5 = v5_1 - v5_2 - v5_3 + v5_4 - v5_5 + v5_6

            v6_1 = 1 / ((self.l3**6) * (6 * self.l3 + 1))
            v6_2 = 1 / ((self.l4**6) * (6 * self.l4 + 1))
            v6_3 = 6 * beta_function(5 * self.l3 + 1, self.l4 +
                                     1) / (self.l3**5 * self.l4)
            v6_4 = 15 * beta_function(4 * self.l3 + 1, 2 *
                                      self.l4 + 1) / (self.l3**4 * self.l4**2)
            v6_5 = 20 * beta_function(3 * self.l3 + 1, 3 *
                                      self.l4 + 1) / (self.l3**3 * self.l4**3)
            v6_6 = 15 * beta_function(2 * self.l3 + 1, 4 *
                                      self.l4 + 1) / (self.l3**2 * self.l4**4)
            v6_7 = 6 * beta_function(self.l3 + 1, 5 * self.l4 +
                                     1) / (self.l3 * self.l4**5)
            v6 = v6_1 + v6_2 - v6_3 + v6_4 - v6_5 + v6_6 - v6_7

            a6 = (v6 - 6 * v5 * v1 + 15 * v4 * v1**2 - 20 * v3 * v1**3 + 15 *
                  v2 * v1**4 - 5 * v1**6) / ((v2 - v1**2)**3.00)  # 6th moment

        a4 = (v4 - 4 * v1 * v3 + 6 * (v1**2) * v2 - 3 * v1**4) / \
            ((v2 - v1**2)**2.00)  # Kurtosis

        a3 = (v3 - 3 * v1 * v2 + 2 * v1**3) / ((v2 - v1**2)**1.500)  # Skew

        a1 = self.l1 - (1 / self.l2) * ((1 / (self.l3 + 1)) - (1 / (self.l4 + 1)))  # Mean

        a2 = (v2 - v1**2) / (self.l2**2)  # Variance

        if return_6th:
            return a1, a2, a3, a4, a6
        else:
            return a1, a2, a3, a4

    def get_samples(self, n_sample=1000):

        F_vals = np.linspace(0, 1, n_sample)
        x_vals = []
        for item in F_vals:
            x_vals.append(self.quantile_function(item))

        x_vals = np.array(x_vals)

        # np.logical_and(x_vals>-1.2e20, x_vals<1.2e20)
        trimming_mask = np.isfinite(x_vals)

        F_vals = F_vals[trimming_mask]
        x_vals = x_vals[trimming_mask]

        return x_vals

    def pdf_value(self, u):
        '''
        Density quantile function
        '''
        temp = (u**(self.l3 - 1) + (1 - u)**(self.l4 - 1)) / self.l2
        return (1 / temp)

    def pdf_plot(self, n_sample=500, ax=None, return_ax=True):

        F_vals = np.linspace(0, 1, n_sample)
        x_vals = list(map(self.quantile_function, F_vals))
        pdf_vals = list(map(self.pdf_value, F_vals))

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        ax.plot(x_vals, pdf_vals, c='g', lw=3, label='Fitted Lambda distribution')

        if return_ax:
            return ax

    def dist_plot(self, n_sample=1000, cdf=True, color='b', lw=2, ax=None, return_samples=False, check_6th=False, stat_xy=(1.01, 0.8), legend=True):

        if check_6th:
            mean, variance, skew, kurtosis, moment_6 = self.get_moments(return_6th=True)
        else:
            mean, variance, skew, kurtosis = self.get_moments()

        F_vals = np.linspace(0, 1, n_sample)
        x_vals = []
        x_vals = list(map(self.quantile_function, F_vals))

        x_vals = np.array(x_vals)

        # np.logical_and(x_vals>-1.2e20, x_vals<1.2e20)
        trimming_mask = np.isfinite(x_vals)

        F_vals = F_vals[trimming_mask]
        x_vals = x_vals[trimming_mask]

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        if cdf:
            ax.plot(x_vals, F_vals, c=color, lw=lw, label='Lambda Fit')
            ax.set_ylabel('CDF')
        else:
            _ = gs.histogram_plot(x_vals, ax=ax, stat_blk=False,
                                  label='Lambda Fit', color=color)

        if check_6th:
            ax.text(stat_xy[0], stat_xy[1],
                    'Mean: {mean:.3f} \n $\sigma: {sigma:.3f}$ \n skew: {skew: .3f} \n kurtosis: {kurtosis:.3f} \n moment_6: {moment_6:.3f}'.format(
                        mean=mean, sigma=np.sqrt(variance), skew=skew, kurtosis=kurtosis, moment_6=moment_6),
                    horizontalalignment='left',
                    verticalalignment='center',
                    transform=ax.transAxes)
        else:
            ax.text(stat_xy[0], stat_xy[1],
                    'Mean: {mean:.3f} \n $\sigma: {sigma:.3f}$ \n skew: {skew: .3f} \n kurtosis: {kurtosis:.3f}'.format(
                        mean=mean, sigma=np.sqrt(variance), skew=skew, kurtosis=kurtosis),
                    horizontalalignment='left',
                    verticalalignment='center',
                    transform=ax.transAxes)

        mean_calc = np.mean(x_vals)
        std_calc = np.std(x_vals)
        skew_calc = np.mean((x_vals - mean_calc)**3) / std_calc**3
        kurtosis_calc = np.mean((x_vals - mean_calc)**4) / std_calc**4
        moment_6_calc = np.mean((x_vals - mean_calc)**6) / std_calc**6

        if check_6th:
            ax.text(stat_xy[0], stat_xy[1] - 0.4,
                    'Calculated: \n Mean: {mean:.3f} \n $\sigma: {sigma:.3f}$ \n skew: {skew: .3f} \n kurtosis: {kurtosis:.3f} \n moment_6: {moment_6:.3f}'.format(
                        mean=mean_calc, sigma=std_calc, skew=skew_calc, kurtosis=kurtosis_calc, moment_6=moment_6_calc),
                    horizontalalignment='left',
                    verticalalignment='center',
                    transform=ax.transAxes)
        else:
            ax.text(stat_xy[0], stat_xy[1] - 0.4,
                    'Calculated: \n Mean: {mean:.3f} \n $\sigma: {sigma:.3f}$ \n skew: {skew: .3f} \n kurtosis: {kurtosis:.3f}'.format(
                        mean=mean_calc, sigma=std_calc, skew=skew_calc, kurtosis=kurtosis_calc),
                    horizontalalignment='left',
                    verticalalignment='center',
                    transform=ax.transAxes)

        ax.grid()
        if legend:
            _ = ax.legend()

        if return_samples:
            return x_vals

    @classmethod
    def normal_dist_comparison(cls, lambda1, lambda2, n_sample=10000, ax=None, fontsize=14):

        lambda_vals = [np.float64(lambda1), np.float64(
            lambda2), np.float64(0.13490936), np.float64(0.13490936)]

        lambda_dist = cls(*lambda_vals)

        mean_lambda, variance_lambda, skew_lambda, kurtosis_lambda = lambda_dist.get_moments()

        F_vals = np.linspace(0, 1, n_sample)
        x_vals = []
        y_vals = []

        for item in F_vals:
            x_vals.append(lambda_dist.quantile_function(item))
            y_vals.append(normal_dist.ppf(item, loc=0, scale=1))  # percent point function

        mean_calc = np.mean(x_vals)

        variance_calc = np.var(x_vals)

        std_calc = np.std(x_vals)

        skew_calc = np.mean((x_vals - mean_calc)**3) / std_calc**3

        kurtosis_calc = np.mean((x_vals - mean_calc)**4) / std_calc**4

        print('Mean: lambda distribution={:.3f}, sampled data={:.3f}, target={:.3f}'.format(
            mean_lambda, mean_calc, 0))

        print('Variance: lambda distribution={:.3f}, sampled data={:.3f}, target={:.3f}'.format(
            variance_lambda, variance_calc, 1))

        print('Skewness: lambda distribution={:.3f}, sampled data={:.3f}, target={:.3f}'.format(
            skew_lambda, skew_calc, 0))

        print('Kurtosis: lambda distribution={:.3f}, sampled data={:.3f}, target={:.3f}'.format(
            kurtosis_lambda, kurtosis_calc, 3))

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        ax.plot(x_vals, F_vals, c='b', lw=1.5, label='Fitted Lambda Distribution')
        ax.plot(x_vals, F_vals, c='salmon', lw=2.5,
                label='Standard Normal Distribution', ls=':')
        ax.grid()
        ax.text(1.01, 0.8,
                'Mean: {mean:.3f} \n $\sigma: {sigma:.3f}$ \n skew: {skew: .3f} \n kurtosis: {kurtosis:.3f}'.format(
                    mean=mean_calc, sigma=std_calc, skew=skew_calc, kurtosis=kurtosis_calc),
                horizontalalignment='left',
                verticalalignment='center',
                transform=ax.transAxes)

        ax.text(1.01, 0.5,
                '$\lambda_1$ = {l1:.3f} \n $\lambda_2$ = {l2:.3f} \n $\lambda_3$ = {l3:.3f} \n $\lambda_4$ = {l4:.3f}'.format(
                    l1=lambda_dist.l1, l2=lambda_dist.l2, l3=lambda_dist.l3, l4=lambda_dist.l4),
                horizontalalignment='left',
                verticalalignment='center',
                transform=ax.transAxes)
        ax.set_ylabel('CDF')
        _ = ax.legend(fontsize=fontsize)

    def simulate(self):
        '''
        Returns a single simulated value
        '''
        u = np.random.rand()

        return self.quantile_function(u)

    @classmethod
    def lambda3_4_skew_kurtosis(cls, l3_range, l4_range, n_mesh=100, make_flat=False):

        l3_mesh, l4_mesh = np.meshgrid(np.linspace(*l3_range, n_mesh),
                                       np.linspace(*l4_range, n_mesh))

        skew_surf = np.zeros(l3_mesh.shape)
        kurtosis_surf = np.zeros(l3_mesh.shape)

        for i in range(l3_mesh.shape[0]):
            for j in range(l3_mesh.shape[1]):
                _, _, a3, a4 = cls(0, 0, l3_mesh[i, j], l4_mesh[i, j]).get_moments()
                skew_surf[i, j] = a3
                kurtosis_surf[i, j] = a4

        if make_flat:
            return l3_mesh.flatten(), l4_mesh.flatten(), skew_surf.flatten(), kurtosis_surf.flatten()
        else:
            return l3_mesh, l4_mesh, skew_surf, kurtosis_surf

    @classmethod
    def lambda34_moments(cls, l3_range, l4_range, n_mesh=100):

        l3_mesh, l4_mesh = np.meshgrid(np.linspace(*l3_range, n_mesh),
                                       np.linspace(*l4_range, n_mesh))

        skew_surf = np.zeros(l3_mesh.shape)
        kurtosis_surf = np.zeros(l3_mesh.shape)
        moment_6_surf = np.zeros(l3_mesh.shape)
        for i in range(l3_mesh.shape[0]):
            for j in range(l3_mesh.shape[1]):
                _, _, a3, a4, a6 = cls(
                    0, 0, l3_mesh[i, j], l4_mesh[i, j]).get_moments(return_6th=True)
                skew_surf[i, j] = a3
                kurtosis_surf[i, j] = a4
                moment_6_surf[i, j] = a6

        return l3_mesh.flatten(), l4_mesh.flatten(), skew_surf.flatten(), kurtosis_surf.flatten(), moment_6_surf.flatten()

    @staticmethod
    def lambda_class_plot():
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        ax.plot([-8, 10], [1, 1], ls='--', lw=3, c='k')
        ax.plot([1, 10], [2, 2], ls='--', lw=3, c='k')
        ax.plot([1, 1], [-10, 10], ls='--', lw=3, c='k')
        ax.plot([2, 2], [1, 10], ls='--', lw=3, c='k')
        ax.plot([0, 5], [0, 5], lw=1.5, c='g')
        ax.set_xlim([0, 5])
        ax.set_ylim([0, 5])
        ax.set_xlabel(r'$\lambda_3$', fontsize=16)
        ax.set_ylabel(r'$\lambda_4$', fontsize=16)
        ax.text(0.45, 0.16, '$I$', fontsize=16, color='b')
        ax.text(3, 0.4, '$II$', fontsize=16, color='b')
        ax.text(1.4, 1.15, '$III$', fontsize=16, color='b')
        ax.text(2.6, 1.3, '$IV$', fontsize=16, color='b')
        _ = ax.text(3.2, 2.5, '$V$', fontsize=16, color='b')


class SampleLambdaMoments(object):

    def __init__(self, n_mesh=500):

        self.n_mesh = n_mesh
        self.class_dict = {'Negative': [[-0.24999, -0.01], [-0.24999, -0.01]],
                           'Ia': [[0.001, 0.55], [0.001, 0.55]],
                           'Ib': [[0.5001, 1], [0.001, 0.5]],
                           'Ic': [[0.5, 1], [0.5, 1]],
                           'II': [[1, 8], [0.001, 1]],
                           'III': [[1, 2], [1, 2]],
                           'IV': [[2, 8], [1, 2]],
                           'V': [[2, 8], [2, 8]]
                           }

    def get_sample(self, class_name='Ia', return_6th=False):

        if class_name not in self.class_dict.keys():
            raise ValueError(
                'Provided class value is not valid based on the following list:')
            for item in self.class_dict.keys():
                print(item)
        if return_6th:
            l3, l4, skew, kurtosis, moment_6 = GeneralizedLambdaDist.lambda34_moments(l3_range=self.class_dict[class_name][0],
                                                                                      l4_range=self.class_dict[class_name][1],
                                                                                      n_mesh=self.n_mesh)
            data_lambda = pd.DataFrame(
                columns=['Skewness', 'Kurtosis', 'moment_6', 'Lambda3', 'Lambda4'])
        else:
            l3, l4, skew, kurtosis = GeneralizedLambdaDist.lambda3_4_skew_kurtosis(l3_range=self.class_dict[class_name][0],
                                                                                   l4_range=self.class_dict[class_name][1],
                                                                                   n_mesh=self.n_mesh, make_flat=True)
            data_lambda = pd.DataFrame(
                columns=['Skewness', 'Kurtosis', 'Lambda3', 'Lambda4'])

        data_lambda['Skewness'] = skew
        data_lambda['Kurtosis'] = kurtosis
        if return_6th:
            data_lambda['moment_6'] = moment_6
        data_lambda['Lambda3'] = l3
        data_lambda['Lambda4'] = l4

        return data_lambda

    @classmethod
    def scatter_plot(cls, n_mesh=20, class_name='Ia'):
        temp_cls = cls(n_mesh=n_mesh)
        data_lambda = temp_cls.get_sample(class_name=class_name)

        init_notebook_mode()
        scene = dict(
            xaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            yaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            zaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.8))
        )

        layout = go.Layout(margin=dict(l=0,
                                       r=0,
                                       b=0,
                                       t=0
                                       ), showlegend=False, title=go.layout.Title(
            text='Summary Visualization',
            font=dict(
                family='Courier New, monospace',
                size=20,
                color='#7f7f7f'
            )
        )
        )

        opacity = 1
        axis_font_size = 16
        axis_font_color = '#000000'

        locationmap_lambda3 = go.Scatter(x=data_lambda.Skewness,
                                         y=data_lambda.Kurtosis,
                                         mode='markers',
                                         marker=dict(size=3, color=data_lambda.Lambda3, colorscale='Jet', showscale=True, colorbar=dict(len=0.3, thickness=10, x=0.4, y=0.15, title='lambda3')
                                                     )
                                         )

        locationmap_lambda4 = go.Scatter(x=data_lambda.Skewness,
                                         y=data_lambda.Kurtosis,
                                         mode='markers',
                                         marker=dict(size=3, color=data_lambda.Lambda4, colorscale='Jet', showscale=True, colorbar=dict(len=0.3, thickness=10, x=0.9, y=0.15, title='lambda4')
                                                     ), xaxis='x2', yaxis='y2'
                                         )

        fig = go.Figure(data=[locationmap_lambda3, locationmap_lambda4], layout=layout)

        fig['layout'].update(title=' ', height=300, width=950)

        layout = layout

        fig['layout']['xaxis'].update({
            "anchor": "y",
            "title": 'Skew',
            "domain": [0.1, 0.4]
        })
        fig['layout']['yaxis'].update({
            "anchor": "x",
            "domain": [0.09, 0.95],
            "title": dict(text='Kurtosis', font=dict(family='Courier New, monospace', size=axis_font_size, color=axis_font_color))
        })

        fig['layout']['xaxis2'] = go.layout.XAxis()
        fig['layout']['xaxis2'].update({
            "anchor": "y2",
            "title": 'Skew',
            "domain": [0.6, 0.9]
        })
        fig['layout']['yaxis2'] = go.layout.YAxis()
        fig['layout']['yaxis2'].update({
            "anchor": "x2",
            "domain": [0.09, 0.95],
            "title": dict(text='Kurtosis', font=dict(family='Courier New, monospace', size=axis_font_size, color=axis_font_color))
        })

        config = {
            'showLink': False,
            'doubleClick': 'reset+autosize',
            'responsive': True,
            'autosizable': True,
            'displayModeBar': True,
            'scrollZoom': False,
            'displayModeBar': True,
            'editable': False}
        _ = iplot(fig, filename='multiple_plots', config=config)


class Moments2Lambdas(object):

    def __init__(self, reference_data, num_nbrs=2):

        self.data = reference_data
        self.skew_column = 'Skewness'
        self.kurtosis_columns = 'Kurtosis'
        self.lambda3_column = 'Lambda3'
        self.lambda4_column = 'Lambda4'
        coordinate_array = np.array(
            [self.data[self.skew_column], self.data[self.kurtosis_columns]]).T
        self.nbrs = NearestNeighbors(
            n_neighbors=num_nbrs, algorithm='kd_tree', metric='minkowski', p=2).fit(coordinate_array)

    def get_lambdas(self, mean, variance, skewness, kurtosis, estimate=False):

        distances, indices = self.nbrs.kneighbors([[skewness, kurtosis]])
        distances = distances.flatten()
        indices = indices.flatten()

        lambda3 = 0
        lambda4 = 0
        w_sum = 0

        if estimate:  # Inverse distance estimate
            if (distances.max() / distances.min()) > 10:
                lambda3 = self.data[self.lambda3_column].loc[indices[0]]
                lambda4 = self.data[self.lambda4_column].loc[indices[0]]
            else:
                # Inverse distance estimate
                for i in range(self.nbrs.n_neighbors):
                    lambda3 += self.data[self.lambda3_column].loc[indices[i]
                                                                  ] * (1 / distances[i])
                    lambda4 += self.data[self.lambda4_column].loc[indices[i]
                                                                  ] * (1 / distances[i])
                    w_sum += (1 / distances[i])
                lambda3 = lambda3 / w_sum
                lambda4 = lambda4 / w_sum
        else:
            lambda3 = self.data[self.lambda3_column].loc[indices[0]]
            lambda4 = self.data[self.lambda4_column].loc[indices[0]]

        v1_1 = lambda3 * (lambda3 + 1)
        v1_2 = lambda4 * (lambda4 + 1)
        v1 = (1 / v1_1) - (1 / v1_2)

        v2_1 = 1 / ((lambda3**2) * (2 * lambda3 + 1))
        v2_2 = 1 / ((lambda4**2) * (2 * lambda4 + 1))
        v2_3 = 2 * beta_function(lambda3 + 1, lambda4 + 1) / (lambda3 * lambda4)
        v2 = v2_1 + v2_2 - v2_3

        lambda2 = np.sqrt((v2 - v1**2) / variance)

        lambda1 = mean + (1 / lambda2) * ((1 / (lambda3 + 1)) - (1 / (lambda4 + 1)))

        return lambda1, lambda2, lambda3, lambda4
