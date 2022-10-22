from scipy.stats import multivariate_normal
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pygeostat as gs
import matplotlib
try:
    from . utility import GoldenSectionSearch
except ImportError:
    from utility import GoldenSectionSearch


def _tickoff(ax, xtickoff, ytickoff):
    '''Remove the xtick and/or ytick labels from the an axis handle'''
    if xtickoff:
        ax.tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False)
        ax.set_xlabel('')
    if ytickoff:
        ax.tick_params(
            axis='y',
            which='both',
            left=False,
            right=False,
            labelleft=False)
        ax.set_ylabel('')


class GmmUtility(object):

    '''
    A class to facilitate data analysis and visualization for Gaussian Mixture Model (GMM).

    Gaussian mixture model is considered an unsupervised machine learning technique to fit the multivariate distribution of observed data.
    GMM is usually fitted based on maximum expectations(EM) and based on maximizing log likelihood of joint distribution of all observations.
    '''

    def __init__(self, gmm_file=None, data=None, variable_names=None, mean_vector_list=None, covariance_matrix_list=None, contribution_list=None):

        if gmm_file is not None and mean_vector_list is not None:
            raise ValueError(
                'Either a gmm file needs to be provided or lists of mean and covariance matrix ')

        # read mixture models from a file (gmm format from CCG program written by Diogo Silva)
        if gmm_file is not None:
            self.gmm_file = gmm_file
            self.__get_mixtures_from_file(self.gmm_file)

        # Process the GMM fitted model from the provided list of mean vectors, covariance matrices and contributions
        if mean_vector_list is not None:
            self.__get_mixtures_from_list(
                mean_vector_list, covariance_matrix_list, contribution_list)

        if variable_names is None:
            self.variable_names = []
            for i in range(self.n_var):
                self.variable_names.append('variable_{:g}'.format(i + 1))
        else:
            if len(variable_names) != self.n_var:
                raise ValueError(
                    'variable_names must have {} parameters'.format(self.n_var))
            else:
                self.variable_names = variable_names

        if data is None:
            self.data = pd.DataFrame(columns=self.variable_names)
        else:
            self.data = data[variable_names]

        if not isinstance(self.data, pd.DataFrame):
            raise ValueError('provided data must be of type pandas dataframe')

    def __get_mixtures_from_list(self, mean_vector_list, covariance_matrix_list, contribution_list):
        '''
        A method to process GMM model and assign the required parameters to the instance of the object
        '''

        if not isinstance(mean_vector_list, list):
            raise ValueError('mean_vector_list must be a list')

        if not isinstance(covariance_matrix_list, list):
            raise ValueError('covariance_matrix_list must be a list')

        if not isinstance(contribution_list, list):
            raise ValueError('contribution_list must be a list')

        self.mean_vectors = []
        self.n_components = len(mean_vector_list)
        try:
            for g in range(self.n_components):
                self.mean_vectors.append(np.array(mean_vector_list[g]))
        except:
            raise ValueError('Each mean vector must be convertable to a numpy array')

        self.n_var = len(self.mean_vectors[0])

        if covariance_matrix_list is None:
            raise ValueError('covariance_matrix_list is required')

        self.cov_matrices = []
        try:
            for g in range(self.n_components):
                self.cov_matrices.append(np.array(covariance_matrix_list[g]))
        except:
            raise ValueError(
                'Each covariance matrix must be convertable to a numpy array')

        if contribution_list is None:
            raise ValueError('covariance_matrix_list is required')

        self.contributions = []
        for g in range(self.n_components):
            self.contributions.append(np.array(contribution_list[g]))

    def __get_mixtures_from_file(self, flname):
        '''
        A method to read the mixture models from an ascii file (CCG program GMM_FIT, Diogo Silva)
        '''
        with open(flname, 'r') as file:
            lines = file.readlines()
        self.n_components = int(lines[1].split()[0])
        self.n_var = int(lines[1].split()[1])

        self.contributions = []
        self.mean_vectors = []
        self.cov_matrices = []
        for i in range(self.n_components):
            contribution = float(lines[i * 3 + 2].split()[1])
            self.contributions.append(contribution)

            mean_vector = np.zeros(self.n_var)
            for j in range(self.n_var):
                mean_vector[j] = float(lines[i * 3 + 3].split()[j])
            self.mean_vectors.append(mean_vector)

            cov_matrix = np.zeros((self.n_var, self.n_var))
            start = 0
            end = self.n_var
            for j in range(self.n_var):
                # lines[i*3+4].split()[j*self.n_var:j*self.n_var+(self.n_var-j)]
                cov_matrix[j, j:] = lines[i * 3 + 4].split()[start:end]
                cov_matrix[j, j] = cov_matrix[j, j] / 2
                start = end
                end = start + (self.n_var - j - 1)
            cov_matrix = (cov_matrix + cov_matrix.T)
            self.cov_matrices.append(cov_matrix)

    def pdf_marginal(self, var_index, x, return_gmm_components=False):
        '''
        A method to calculate marginal univariate and multivariate distributions based on GMM components.
        Note that the var_index matches the index of variables being provided for the GMM algorithm and also should
        match the variable name sequence provided in constructor of the class GmmUtility.
        '''

        if var_index is None:
            var_index = [i for i in range(self.n_var)]

        try:
            var_index = np.array(var_index)
            var_index = var_index.flatten()
        except:
            raise ValueError('x must be convertable to numpy array')

        n_marginal = len(var_index)

        try:
            x = np.array(x)
        except:
            raise ValueError('x must be convertable to numpy array')
        # x = x.reshape(-1,n_marginal)

        output = 0
        mean_list = []
        cov_list = []

        for g in range(self.n_components):
            mean_marginal = self.mean_vectors[g][var_index]
            mean_list.append(mean_marginal)
            covariance_marginal = np.zeros((n_marginal, n_marginal))
            for i, idx_i in enumerate(var_index):
                for j, idx_j in enumerate(var_index):
                    covariance_marginal[i, j] = self.cov_matrices[g][idx_i, idx_j]
            cov_list.append(covariance_marginal)
            output += MultivariateNormal(mean_marginal,
                                         covariance_marginal).pdf(x) * self.contributions[g]

        if return_gmm_components:
            return output, mean_list, cov_list
        else:
            return output

    def summary_plot(self, figsize=None, cmap='viridis', pad=0, cbar=True, return_axes=False):
        '''
        A method to provide summary univariate and bivariate distributions for GMM fitted model along with the provided data points.
        '''

        if figsize is None:
            figsize = (self.n_var * 5, self.n_var * 4)
        fig, axes = plt.subplots(self.n_var, self.n_var, figsize=figsize)

        for i in range(self.n_var):
            for j in range(self.n_var):
                if i < j:
                    plot, levels = self.__bivariate_plot(var_index=np.array(
                        [j, i]), cmap=cmap, ax=axes[i, j], cbar_label=False, cbar=False)
                    if i == j - 1:
                        _tickoff(axes[i][j], xtickoff=True, ytickoff=False)
                    else:
                        _tickoff(axes[i][j], xtickoff=True, ytickoff=True)
                elif i == j:
                    self.__univariate_plot(var_index=i, ax=axes[i, j], legend=True)
                else:
                    axes[i, j].axis('off')

        if cbar:
            cbar_ax = fig.add_axes([0.2, .15, .03, .25])
            cbar = fig.colorbar(plot, cax=cbar_ax,
                                ticks=np.linspace(levels[0], levels[-1], 3))
            cbar.set_label('PDF', ha='center', va='top', labelpad=2, fontsize=22)
            cbar.ax.set_yticklabels(['Low', 'Med.', 'High'], fontsize=20)

        try:
            fig.tight_layout(h_pad=pad[1], w_pad=pad[0])
        except:
            fig.tight_layout(h_pad=pad, w_pad=pad)
        if return_axes:
            return axes

    def __bivariate_plot(self, var_index, s=80, scatter=True, cmap='viridis', ax=None, figsize=(6, 6), clim=None, sigfigs=None, kernel_lower_percentile=50, cbar=True, cbar_label=True):
        '''
        A method for bivariate plotting of marginal GMMs
        '''

        if not isinstance(var_index, np.ndarray):
            raise ValueError('va_index mus be a numpy array with length 2')

        if (len(var_index) != 2):
            raise ValueError('var_index must have two elements')

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            if cbar:
                cax = None
                fig, ax, cbar_ax = gs.setup_plot(ax, cax=cax, cbar=True, figsize=figsize)

        xmin = np.min(self.data[self.variable_names[var_index[0]]])
        xmax = np.max(self.data[self.variable_names[var_index[0]]])

        ymin = np.min(self.data[self.variable_names[var_index[1]]])
        ymax = np.max(self.data[self.variable_names[var_index[1]]])

        x, y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y
        kernel = self.pdf_marginal(var_index=var_index, x=pos)

        # scatter points on the main axes
        if scatter:
            ax.scatter(self.data[self.variable_names[var_index[0]]],
                       self.data[self.variable_names[var_index[1]]], s=s, facecolors='none', edgecolors='gray')

        lvs = np.linspace(np.percentile(
            kernel.ravel(), kernel_lower_percentile), np.max(kernel.ravel()), 10)
        contour = ax.contour(x, y, kernel, cmap=cmap, levels=lvs)
        contourf = ax.contourf(x, y, kernel, alpha=0.75, cmap=cmap, levels=lvs)

        ax.set_xlabel(self.variable_names[var_index[0]])
        ax.set_ylabel(self.variable_names[var_index[1]])

        if cbar:
            cbar = fig.colorbar(contourf, ticks=lvs, format='%.3f', cax=cbar_ax)
            if cbar_label:
                cbar.set_label('pdf', ha='center', va='top', labelpad=2)

        return contourf, lvs

    def __univariate_plot(self, var_index, ax=None, invert_axes=False, figsize=(6, 6), legend=False, add_label=True):
        '''
        A method for univariate pdf plot of univariate marginal and conditional distributions.
        '''
        try:
            var_index = int(var_index)
        except:
            raise ValueError('var_index must be an integer')

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        xmin = np.min(self.data[self.variable_names[var_index]])
        xmax = np.max(self.data[self.variable_names[var_index]])

        x = np.linspace(xmin, xmax, 100).reshape(100, 1)

        pdf, mean_list, cov_list = self.pdf_marginal(
            var_index, x, return_gmm_components=True)
        if invert_axes:
            ax.plot(pdf, x, c='b', lw=3, ls='--', label='Fitted GMM')
            for g in range(self.n_components):
                pdf_gmm = MultivariateNormal(mean_list[g], cov_list[g]).pdf(
                    x) * self.contributions[g]
                if g == 0:
                    ax.plot(pdf_gmm, x, c='darkorange', lw=1.5, label='GMM components')
                else:
                    ax.plot(pdf_gmm, x, c='darkorange', lw=1.5)

            mask = pd.isnull(self.data[self.variable_names[var_index]])
            ax.hist(self.data[~mask][self.variable_names[var_index]], density=True, bins=20,
                    orientation='horizontal', facecolor='gray', alpha=0.75, edgecolor='k', label='Data')

        else:
            ax.plot(x, pdf, c='b', lw=3, ls='--', label='Fitted GMM')
            for g in range(self.n_components):
                pdf_gmm = MultivariateNormal(mean_list[g], cov_list[g]).pdf(
                    x) * self.contributions[g]
                if g == 0:
                    ax.plot(x, pdf_gmm, c='darkorange', lw=1.5, label='GMM components')
                else:
                    ax.plot(x, pdf_gmm, c='darkorange', lw=1.5)

            mask = pd.isnull(self.data[self.variable_names[var_index]])
            ax.hist(self.data[~mask][self.variable_names[var_index]], density=True, bins=20,
                    orientation='vertical', facecolor='gray', alpha=0.75, edgecolor='k', label='Data')
        if legend:
            ax.legend(loc=2, fontsize=14)

        if add_label:
            ax.set_xlabel(self.variable_names[var_index])
            ax.set_ylabel('pdf')

    def bivariate_plot(self, var_index, cmap='viridis', figsize=(8, 8)):
        '''
        A method to provide a grided plot of bivariate and univariate.
        '''

        try:
            var_index = np.array(var_index)
            var_index = var_index.flatten()
        except:
            raise ValueError('index list must be convertable to numpy array')

        if (len(var_index) != 2):
            raise ValueError('var_index must have two elements')

        # Set up the axes with gridspec
        fig = plt.figure(figsize=figsize)
        grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.5)
        main_ax = fig.add_subplot(grid[:-1, 1:])
        y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
        x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

        self.__bivariate_plot(var_index=var_index, cmap=cmap, ax=main_ax)

        # pdf on the attached axes (1)
        self.__univariate_plot(var_index=var_index[0], ax=x_hist, add_label=False)
        x_hist.invert_yaxis()
        x_hist.set_axis_off()
        x_hist.get_xaxis().set_visible(False)
        x_hist.get_yaxis().set_visible(False)

        # pdf on the attached axes (2)
        self.__univariate_plot(
            var_index=var_index[1], ax=y_hist, invert_axes=True, add_label=False)
        y_hist.invert_xaxis()
        y_hist.set_axis_off()
        y_hist.get_xaxis().set_visible(False)
        y_hist.get_yaxis().set_visible(False)

    @staticmethod
    def get_moments(mean_list, cov_list, contrib_list):
        '''
        A method to calculated univariate moments (i.e. mean, variance, skewness and kurtosis) based on provided list of mean, variance and contributions for
        all mixtures.
        '''

        n_components = len(mean_list)

        if len(cov_list) != n_components:
            raise ValueError(
                'list of covariance for Gaussian components need to match the number of components ({})'.format(n_components))

        if len(contrib_list) != n_components:
            raise ValueError(
                'list of contribution factors for Gaussian components need to match the number of components ({})'.format(n_components))

        mu_m = 0
        for g in range(n_components):

            if len(mean_list[g]) > 1 or len(cov_list[g].flatten()) > 1:
                raise ValueError(
                    'Moments are avaliable just for univariate mixture models')

            mu_m += contrib_list[g] * mean_list[g][0]

        var_m = 0
        for g in range(n_components):
            var_m += contrib_list[g] * (cov_list[g][0, 0] + mean_list[g]
                                        [0]**2 - 2 * mean_list[g][0] * mu_m + mu_m**2)

        skewness_m = 0
        for g in range(n_components):
            skewness_m += contrib_list[g] * ((mean_list[g][0]**3 + 3 * mean_list[g][0] * cov_list[g][0, 0]) - (
                3 * (cov_list[g][0, 0] + mean_list[g][0]**2) * mu_m) + (3 * mean_list[g][0] * mu_m**2) - mu_m**3)

        skewness_m = skewness_m / (var_m**(1.5000000))

        kurtosis_m = 0
        for g in range(n_components):
            # kurtosis_m += contrib_list[g] * ( mean_list[g][0]**4 + 6*mean_list[g][0]*cov_list[g][0,0] + 3*cov_list[g][0,0]**2  -4*(mean_list[g][0]**3 + 3*mean_list[g][0]*cov_list[g][0,0])*mu_m + 6*(cov_list[g][0,0] + mean_list[g][0]**2)*mu_m**2 - 4*mean_list[g][0]*mu_m**3 +mu_m**4)
            kurtosis_m += contrib_list[g] * ((mean_list[g][0]**4 + 6 * mean_list[g][0]**2 * cov_list[g][0, 0] + 3 * cov_list[g][0, 0]**2) - 4 * (
                mean_list[g][0]**3 + 3 * mean_list[g][0] * cov_list[g][0, 0]) * mu_m + 6 * (cov_list[g][0, 0] + mean_list[g][0]**2) * mu_m**2 - 4 * (mean_list[g][0] * mu_m**3) + mu_m**4)

        kurtosis_m = kurtosis_m / (var_m**2.000000)

        return mu_m, var_m, skewness_m, kurtosis_m

    def conditional_moments(self, conditioning_data):
        '''
        Get conditional moments
        '''

        mean_list, cov_list, contrib_list = self.get_conditional_pdf(conditioning_data)

        mu_m, var_m, skewness_m, kurtosis_m = GmmUtility.get_moments(
            mean_list, cov_list, contrib_list)

        return mu_m, var_m, skewness_m, kurtosis_m

    def univariate_conditional_plot(self, conditioning_data, legend=True, return_moments=False, axes=None, cdf=True):
        '''
        A method to plot univariate conditional PDF and CDF based on GMM contributions, conditional means and variances
        '''

        if axes is None:
            if cdf:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            else:
                fig, axes = plt.subplots(1, 1, figsize=(6, 4))
                axes = [axes]

        x_pdf, conditional_pdf, mean_list, cov_list, contrib_list = self.univariate_conditional_pdf(
            conditioning_data, x=None, return_gmm_components=True)
        if cdf:
            x_cdf, conditional_cdf = self.univariate_conditional_cdf(
                conditioning_data, x=None)

        mu_m, var_m, skewness_m, kurtosis_m = GmmUtility.get_moments(
            mean_list, cov_list, contrib_list)
        sigma_m = np.sqrt(var_m)

        axes[0].plot(x_pdf, conditional_pdf, c='k', lw=3, label='Fitted GMM')
        for g in range(self.n_components):
            pdf_gmm = MultivariateNormal(
                mean_list[g], cov_list[g]).pdf(x_pdf) * contrib_list[g]
            if g == 0:
                axes[0].plot(x_pdf, pdf_gmm, c='darkorange',
                             lw=1.5, label='GMM components')
            else:
                axes[0].plot(x_pdf, pdf_gmm, c='darkorange', lw=1.5)
        if legend:
            axes[0].legend(loc=2)

        if cdf:
            axes[1].plot(x_cdf, conditional_cdf, c='k', lw=3)

        if cdf:
            ax = axes[1]
        else:
            ax = axes[0]
        ax.text(0.7, 0.85,
                'Mean: {mean:.3f} \n $\sigma: {sigma:.3f}$ \n skew: {skew: .3f} \n kurtosis: {kurtosis:.3f}'.format(
                    mean=mu_m, sigma=sigma_m, skew=skewness_m, kurtosis=kurtosis_m),
                horizontalalignment='left',
                verticalalignment='center',
                transform=ax.transAxes)

        axes[0].set_ylabel('PDF')
        if cdf:
            axes[1].set_ylabel('CDF')
        conditioning_data = np.array(conditioning_data)
        idx_m = np.where(conditioning_data == None)[0][0]
        axes[0].set_xlabel(self.variable_names[idx_m])
        if cdf:
            axes[1].set_xlabel(self.variable_names[idx_m])

        if return_moments:
            return mu_m, var_m, skewness_m, kurtosis_m

    def univariate_conditional_pdf(self, conditioning_data, x=None, return_gmm_components=False):
        '''
        A method to calculate univariated conditional pdf for the fitted GMM and based on the provided conditioning data.
        '''

        conditional_means_list, conditional_covariance_list, conditional_contribution_list = self.get_conditional_pdf(
            conditioning_data)

        try:
            conditioning_data = np.array(conditioning_data)
        except:
            raise ValueError('conditioning_data must be convertable to numpy array')

        # index for missing data
        idx_m = np.where(conditioning_data == None)[0]
        n_missing = len(idx_m)

        # This section makes sure that the output will be univariate
        if n_missing > 1:
            raise ValueError(
                'This method is designed to provide univariate conditional pdf')

        return_x = False
        if x is None:
            return_x = True
            xmin = np.min(self.data[self.variable_names[idx_m[0]]])
            xmax = np.max(self.data[self.variable_names[idx_m[0]]])
            x = np.linspace(xmin, xmax, 100).reshape(100, 1)
        try:
            x = np.array(x)
        except:
            raise ValueError('x must be convertable to numpy array')

        if len(x.shape) > 1:
            x = x.flatten()

        x = x.reshape(-1, n_missing)

        output = 0

        for i in range(self.n_components):
            output += MultivariateNormal(conditional_means_list[i], conditional_covariance_list[i]).pdf(
                x) * conditional_contribution_list[i]

        if return_gmm_components:
            if return_x:
                return x, output, conditional_means_list, conditional_covariance_list, conditional_contribution_list
            else:
                return output, conditional_means_list, conditional_covariance_list, conditional_contribution_list
        else:
            if return_x:
                return x, output
            else:
                return output

    def get_conditional_pdf(self, conditioning_data):
        '''
        A method to calculate conditional pdf for the fitted GMM and based on the provided conditioning data.
        '''

        try:
            conditioning_data = np.array(conditioning_data)
        except:
            raise ValueError('conditioning_data must be convertable to numpy array')

        if len(conditioning_data.shape) > 1:
            conditioning_data = conditioning_data.flatten()
        if len(conditioning_data) != self.n_var:
            raise ValueError(
                'conditioning_data has the wrong length. Correct length is{:g}'.format(self.n_var))

        # index for missing data
        idx_m = np.where(conditioning_data == None)[0]
        n_missing = len(idx_m)

        # index for conditional data
        idx_o = np.where(conditioning_data != None)[0]
        conditioning_data = conditioning_data[idx_o].astype(float)
        n_conditional = len(idx_o)

        # get the conditional means for the GMM
        conditional_means_list = []
        conditional_covariance_list = []
        conditional_contribution_list = []
        for g in range(self.n_components):

            # covariance between missing and observed
            cov_mo = np.zeros((n_missing, n_conditional))
            for i, idx_i in enumerate(idx_m):
                for j, idx_j in enumerate(idx_o):
                    cov_mo[i, j] = self.cov_matrices[g][idx_i, idx_j]

            # Covariance between observed data (conditionals)
            cov_oo = np.zeros((n_conditional, n_conditional))
            for i, idx_i in enumerate(idx_o):
                for j, idx_j in enumerate(idx_o):
                    cov_oo[i, j] = self.cov_matrices[g][idx_i, idx_j]

            # Covariance between missing data
            cov_mm = np.zeros((n_missing, n_missing))
            for i, idx_i in enumerate(idx_m):
                for j, idx_j in enumerate(idx_m):
                    cov_mm[i, j] = self.cov_matrices[g][idx_i, idx_j]

            cov_oo_inv = np.linalg.inv(cov_oo)

            # Mean vector for each contribution
            mean_vector_m = self.mean_vectors[g][idx_m]
            mean_vector_o = self.mean_vectors[g][idx_o]

            conditional_means_list.append(
                mean_vector_m + np.matmul(np.matmul(cov_mo, cov_oo_inv), (conditioning_data - mean_vector_o)))
            conditional_covariance_list.append(
                cov_mm - np.matmul(np.matmul(cov_mo, cov_oo_inv), cov_mo.T))

            conditional_contribution_list.append(self.contributions[g] * MultivariateNormal(
                mean_vector=mean_vector_o, cov_matrix=cov_oo).pdf(conditioning_data))

        conditional_contribution_list = conditional_contribution_list / \
            sum(conditional_contribution_list)

        return conditional_means_list, conditional_covariance_list, conditional_contribution_list

    def univariate_conditional_cdf(self, conditioning_data, x):

        try:
            conditioning_data = np.array(conditioning_data)
        except:
            raise ValueError('conditioning_data must be convertable to numpy array')

        # index for missing data
        idx_m = np.where(conditioning_data == None)[0]
        n_missing = len(idx_m)
        if (n_missing != 1):
            raise ValueError(
                'This method is designed to provide univariate conditional cdf')

        return_x = False
        if x is None:
            return_x = True
            xmin = np.min(self.data[self.variable_names[idx_m[0]]])
            xmax = np.max(self.data[self.variable_names[idx_m[0]]])
            x = np.linspace(xmin, xmax, 100).reshape(100, 1)
            cdf = np.zeros(len(x))
        else:
            try:
                x = np.array(x)
                x = x.flatten()
            except:
                raise ValueError('x must be convertable to numpy array')
            cdf = np.zeros(len(x))

        dx = x[1] - x[0]

        cdf_val = 0
        for i, item in enumerate(self.univariate_conditional_pdf(conditioning_data, x)):
            cdf_val += item * dx
            cdf[i] = cdf_val

        if return_x:
            return x, cdf
        else:
            return cdf

    @staticmethod
    def univariate_pdf_from_mixture_plot(mean_list, covariance_list, contribution_list,
                                         variable_name, ax=None, legend=True,
                                         print_moments=False):
        '''
        A method to plot univariate pdf based on the mixtire info including list of mean values, covariance matrices and contributions
        '''
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        n_components = len(mean_list)
        x_pdf, pdf = GmmUtility.univariate_pdf_from_mixture(
            n_components, mean_list, covariance_list, contribution_list, return_x=True)

        ax.plot(x_pdf, pdf, c='k', lw=3, label='Fitted GMM')
        for g in range(n_components):
            pdf_gmm = MultivariateNormal(mean_list[g], covariance_list[g]).pdf(
                x_pdf) * contribution_list[g]
            if g == 0:
                ax.plot(x_pdf, pdf_gmm, c='darkorange', lw=1.5, label='GMM components')
            else:
                ax.plot(x_pdf, pdf_gmm, c='darkorange', lw=1.5)
        if legend:
            ax.legend(loc=2, fontsize=12)

        ax.set_ylabel('PDF', fontsize=12)
        ax.set_xlabel(variable_name, fontsize=12)

        mu_m, var_m, skewness_m, kurtosis_m = GmmUtility.get_moments(
            mean_list, covariance_list, contribution_list)
        sigma_m = np.sqrt(var_m)

        if print_moments:
            print(mu_m, var_m, skewness_m, kurtosis_m)

        ax.text(0.7, 0.85,
                'Mean: {mean:.3f} \n $\sigma: {sigma:.3f}$ \n skew: {skew: .3f} \n kurtosis: {kurtosis:.3f}'.format(
                    mean=mu_m, sigma=sigma_m, skew=skewness_m, kurtosis=kurtosis_m),
                horizontalalignment='left',
                verticalalignment='center',
                transform=ax.transAxes)

    @staticmethod
    def univariate_pdf_from_mixture(n_components, mean_list, covariance_list, contribution_list, x_range=[-4, 4], return_x=True):
        '''
        A method to calculate univariate pdf based on the mixtire info including list of mean values, covariance matrices and contributions
        '''

        x = np.linspace(*x_range, 100).reshape(100, 1)
        output = 0
        for i in range(n_components):

            if len(mean_list[i]) > 1 or len(covariance_list[i].flatten()) > 1:
                raise ValueError(
                    'Moments are avaliable just for univariate mixture models')

            output += MultivariateNormal(mean_list[i],
                                         covariance_list[i]).pdf(x) * contribution_list[i]

        if return_x:
            return x, output
        else:
            return output

    @staticmethod
    def get_modality_measure(mean_list, variance_list, contribution_list, n_increments=1000):
        '''
        A static method to return number of modes and a measure of modality based on a brute force numerical approach for Gaussian distributions
        '''
        def get_density(x):
            output = 0
            x = np.array([x])
            for i in range(len(mean_list)):
                output += MultivariateNormal(mean_list[i],
                                             variance_list[i]).pdf(x) * contribution_list[i]
            return output

        x_array = np.linspace(-4, 4, n_increments)
        n_slope_change = 0
        n_modes = 1
        increment = 1
        modality_measure = 0
        tracking_list = []
        for i in range(n_increments - 1):
            density_b = get_density(x_array[i])
            density_a = get_density(x_array[i + increment])
            if density_a < density_b:
                n_slope_change += 1
                increment *= -1
                n_modes = n_modes + int((1 + increment) / 2)
                tracking_list.append([density_b, x_array[i]])

        for i in range(len(tracking_list) - 1):
            modality_measure += abs(tracking_list[i + 1][0] - tracking_list[i][0]) * (
                tracking_list[i + 1][1] - tracking_list[i][1])

        return n_modes, modality_measure

    # def get_modality_measure(self, mean_list, variance_list, contribution_list):
        # '''
        # A function to calculate the modality measure and number of modes
        # returns modality measure and number of modes
        # '''

        # def get_density(x):
        # 	output = 0
        # 	x = np.array([x])
        # 	for i in range(self.n_components):
        # 		output += MultivariateNormal(mean_list[i], variance_list[i]).pdf(x) * contribution_list[i]
        # 	return output

        # means_sorted = sorted(np.array(mean_list).reshape(self.n_components))
        # modality_measure = 0;
        # n_modes = 1
        # gss = GoldenSectionSearch()
        # for i in range(self.n_components-1):

        # 	a = means_sorted[i]
        # 	b = means_sorted[i+1]

        # 	distance = b-a

        # 	min_f = get_density(gss.calc(get_density,a,b))

        # 	delta1 = abs(get_density(b)-min_f)
        # 	delta2 = abs(get_density(a)-min_f)

        # 	delta = min(delta1,delta2)
        # 	if delta > 0.00001:
        # 		n_modes+=1
        # 		modality_measure += abs(delta1-delta2) * distance

        # 	# modality_measure += delta * distance

        # return modality_measure, n_modes


class UnivariateNormal(object):

    '''
    A class to calculate univariate normal distribution statistics
    '''

    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance

    def pdf(self, x):
        try:
            x = np.array(x)
        except:
            raise ValueError('observations(x) must be convertable to numpy array')

        # output = np.zeros(x.shape)
        # for i, item in enumerate(x):
        # 	output[i] = self.__pdf(item)

        # Using map
        output = np.array(list(map(self.__pdf, x)))
        return output.reshape(x.shape)

    def __pdf(self, x):

        pi = np.pi
        denominator = np.sqrt((2 * pi) * self.variance)
        squared_stat_distance = (x - self.mean)**2 / (2 * self.variance)

        return np.exp(-squared_stat_distance) / denominator


class MultivariateNormal(object):

    '''
    A class to calculate multivariate distribution statistics
    '''

    def __init__(self, mean_vector, cov_matrix):

        try:
            self.mean_vector = np.array(mean_vector)
        except:
            raise ValueError('Mean vector must be convertable to numpy array')

        self.n_d = mean_vector.flatten().shape[0]

        try:
            self.cov_matrix = np.array(cov_matrix)
        except:
            raise ValueError('Covariance matrix must be convertable to numpy array')

    def pdf(self, x):
        try:
            x = np.array(x)
        except:
            raise ValueError('observations(x) must be convertable to numpy array')

        if (x.shape[-1] != self.n_d):
            raise ValueError('The provided tensor x has wrong dimension')

        original_shape = x.shape

        x = x.reshape(-1, self.n_d)

        output = []
        for i in range(x.shape[0]):
            output.append(self.__pdf(x[i, :]))

        return np.array(output).reshape(original_shape[0:-1])

    def __pdf(self, x):

        pi = np.pi

        det_cov = np.linalg.det(self.cov_matrix)
        denominator = np.sqrt((2.0000 * pi)**self.n_d * det_cov)
        cov_matrix_inv = np.linalg.inv(self.cov_matrix)
        squared_stat_distance = np.matmul(
            np.matmul((x - self.mean_vector).T, cov_matrix_inv), (x - self.mean_vector))

        return np.exp(-0.50000 * squared_stat_distance) / denominator
