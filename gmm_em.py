def log_likelihood(N, K, mus, covs, z, X):
    from scipy.stats import multivariate_normal
    outer_sum = 0
    for n in range(N):    
        inner_sum = 0
        for k in range(K):
            inner_sum += z[k] * multivariate_normal.pdf(X[n], mus[k], covs[k])
        outer_sum += inner_sum
    return outer_sum

def e_step(N, K, mus, covs, z, X):
    from scipy.stats import multivariate_normal
    import numpy as np
    responsibilities = np.zeros((N, K))
    # E step
    for n in range(N):
        denom = 0
        for k in range(K):
            denom += z[k] * multivariate_normal.pdf(X[n], mus[k], covs[k])
        for k in range(K):
            pdf_cur = multivariate_normal.pdf(X[n], mus[k], covs[k])
            resp_nk = (z[k] * pdf_cur) / denom
            responsibilities[n, k] = resp_nk
    return responsibilities

def m_step(N, K, X, responsibilities):
    import numpy as np
    NK = np.sum(responsibilities, axis=0) # axis=0 means over rows (n)
    mus = np.zeros((K, 2))
    covs = np.zeros((K, 2, 2))
    z = np.zeros((K, 1))
    for k in range(K):
        mus_inner_sum = 0
        for n in range(N):
            mus_inner_sum += responsibilities[n, k] * X[n]
        mus[k] = mus_inner_sum / NK[k]
        
        covs_inner_sum = 0
        for n in range(N):
            vec = np.reshape((X[n] - mus[k]), (2, 1))
            covs_inner_sum += responsibilities[n, k] * vec.dot(vec.T)
        covs[k] = covs_inner_sum / NK[k]
        z[k] = NK[k] / N
    return mus, covs, z

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    THIS IS DIRECTLY TAKEN FROM "https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html"
    
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
