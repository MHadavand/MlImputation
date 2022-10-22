import math


def save_plot(outfl=None, dpi=300, pad_inches=0.02, **kwargs):
    """
    This function can be used to save plots in different formats (i.e., png, jpg, pdf,...)

    Parameters:
            outfl(str): Output address (directory + file name)
            dpi(int): dpi (dots per inch) for the saved plot
            pad_inches (float): Pad size around the plot
            **kwargs: Optional permissible keyword
    Example:

            >>> RandomColor = PyEDW.save_plot(outfl='E:/Git_Repo/Figure1.pdf', dpi=300, pad_inches =0.02)
    """

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import os

    _, extension = os.path.splitext(outfl)

    if extension.upper() == '.PDF':

        pdfout = PdfPages(outfl)
        plt.savefig(pdfout, format='pdf', bbox_inches='tight',
                    pad_inches=pad_inches, dpi=dpi, **kwargs)
        pdfout.close()
    else:
        plt.savefig(outfl, bbox_inches='tight', pad_inches=pad_inches, dpi=dpi, **kwargs)


def Lambda_visual(l3_mesh, l4_mesh, skew_surf, kurtosis_surf, out_dir=None):
    '''
    Plots skewness and kurtosis vs the shape parameters of lambda distribution (i.e. lambda3 and lambda4)
    '''
    from plotly.offline import iplot, init_notebook_mode
    import plotly.graph_objs as go

    init_notebook_mode(connected=False)
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
                          colorbar=dict(len=0.25, thickness=10, x=0.45, y=0.25),

                          scene='scene1')

    kurtosis = go.Surface(x=l3_mesh, y=l4_mesh, z=kurtosis_surf, colorscale='Viridis', showscale=True,
                          colorbar=dict(len=0.25, thickness=10, x=0.93, y=0.25),
                          scene='scene2')

    fig = go.Figure(data=[skewness, kurtosis], layout=layout)

    fig['layout'].update(title=' ',
                         height=600, width=900)

    layout = layout
    fig['layout']['scene1'] = scene
    fig['layout']['scene2'] = scene
    fig['layout']['scene1']['domain'].update({"x": [0.0, 0.5], "y": [0, 1]})
    fig['layout']['scene2']['domain'].update({"x": [0.5, 1], "y": [0, 1]})
    fig['layout']['scene1']['xaxis'].update({"title": dict(text='lambda_3', font=dict(
        family='Courier New, monospace', size=axis_font_size, color=axis_font_color))})
    fig['layout']['scene1']['yaxis'].update({"title": dict(text='lambda_4', font=dict(
        family='Courier New, monospace', size=axis_font_size, color=axis_font_color))})
    fig['layout']['scene1']['zaxis'].update({"title": dict(text='Skewness', font=dict(
        family='Courier New, monospace', size=axis_font_size, color=axis_font_color))})
    fig['layout']['scene2']['xaxis'].update({"title": dict(text='lambda_3', font=dict(
        family='Courier New, monospace', size=axis_font_size, color=axis_font_color))})
    fig['layout']['scene2']['yaxis'].update({"title": dict(text='lambda_4', font=dict(
        family='Courier New, monospace', size=axis_font_size, color=axis_font_color))})
    fig['layout']['scene2']['zaxis'].update({"title": dict(text='Kurtosis', font=dict(
        family='Courier New, monospace', size=axis_font_size, color=axis_font_color))})

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


class GoldenSectionSearch():

    def __init__(self):

        self.invphi = (math.sqrt(5) - 1) / 2  # 1 / phi
        self.invphi2 = (3 - math.sqrt(5)) / 2  # 1 / phi^2

    def calc(self, f, a, b, tol=1e-9, h=None, c=None, d=None, fc=None, fd=None):
        """ Golden section search, recursive.

        Given a function f with a single local minimum in
        the interval [a,b], gss returns a subset interval
        [c,d] that contains the minimum with d-c <= tol.

        Example:
        >>> f = lambda x: (x-2)**2
        >>> a = 1
        >>> b = 5
        >>> tol = 1e-5
        >>> (c,d) = calc(f, a, b, tol)
        >>> print (c, d)
        (1.9999959837979107, 2.0000050911830893)
        """

        (a, b) = (min(a, b), max(a, b))
        if h is None:
            h = b - a
        if h <= tol:
            return (a + b) / 2
        if c is None:
            c = a + self.invphi2 * h
        if d is None:
            d = a + self.invphi * h
        if fc is None:
            fc = f(c)
        if fd is None:
            fd = f(d)
        if fc < fd:
            return self.calc(f, a, d, tol, h * self.invphi, c=None, fc=None, d=c, fd=fc)
        else:
            return self.calc(f, c, b, tol, h * self.invphi, c=d, fc=fd, d=None, fd=None)


def get_lambdas_keras(mean, variance, skewness, kurtosis, ml_model):
    '''
    A method to calculate lambda parameters for lambda distribution given the four first moments.
    The ml model (keras or tensor flow wrapper) is used to get skewness and kurtosis as features and
    estimate the expected lambda 3 and lambda 4. Lambda 1 and lambda 2 are then calculated mathematically.
    '''
    try:
        from . lambda_distribution import beta_function
    except ImportError:
        from lambda_distribution import beta_function
    import numpy as np

    data = np.array([skewness, kurtosis])
    data = data.reshape(-1, 2)
    temp = ml_model.predict(data)
    lambda3 = temp[0][0]
    lambda4 = temp[0][1]

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


def create_axes(n_col, n_plots, figsize, **kwargs):

    from matplotlib import pyplot as plt

    n_rows = n_plots // n_col + int(n_plots % n_col > 0)

    fig, axes = plt.subplots(n_rows, n_col, figsize=figsize, **kwargs)

    axes = axes.flatten()

    n_invisible = 0
    if (n_plots % n_col) > 0:
        n_invisible = n_col - n_plots % n_col

        for ax in axes[-n_invisible:]:
            ax.set_visible(False)

    return fig, axes[:len(axes) - n_invisible]


def fix_ipython_autocomplete(enable=True):
    r"""Change autocomplete behavior for IPython > 6.x

    Parameter
    ---------
    enable : bool (default True)
        Is use the trick.

    Notes
    -----
    Since IPython > 6.x the ``jedi`` package is using for autocomplete by default.
    But in some cases, the autocomplete doesn't work correctly wrong (see e.g.
    `here <https://github.com/ipython/ipython/issues/11653>`_).

    To set the correct behaviour we should use in IPython environment::

        %config Completer.use_jedi = False

    or add to IPython config (``<HOME>\.ipython\profile_default\ipython_config.py``)::

        c.Completer.use_jedi = False

    """
    try:
        __IPYTHON__
    except NameError:
        pass
    else:
        from IPython import __version__

        major = int(__version__.split(".")[0])
        if major >= 6:
            from IPython import get_ipython

            get_ipython().Completer.use_jedi = not enable


def make_html(notebook, file, environment=None, skip=False):
    from subprocess import call
    if environment is None:
        command_list = [
            "jupyter",
            "nbconvert",
            f"{notebook}",
            "--to",
            "html_toc",
            "--ExtractOutputPreprocessor.enabled=False",
            "--output",
            file
        ]
    else:
        command_list = [
            "conda",
            "activate",
            f"{environment}",
            "&"
            "jupyter",
            "nbconvert",
            f"{notebook}",
            "--to",
            "html_toc",
            "--ExtractOutputPreprocessor.enabled=False",
            "--output",
            file
        ]
    if skip:
        command_list += ['--TagRemovePreprocessor.remove_cell_tags={\"skip\"}']

    res = call(
        command_list,
        shell=True,
    )
    if res != 0:
        raise RuntimeError('Something went wrong')
