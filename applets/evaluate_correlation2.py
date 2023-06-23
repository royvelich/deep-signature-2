import numpy as np

if __name__ == '__main__':
    n = int(1e6)
    # X = np.random.normal(loc=0, scale=3, size=n)
    # Y = np.random.normal(loc=0, scale=3, size=n)
    # Z = np.random.normal(loc=0, scale=3, size=n)

    limit = 1
    rng = np.random.default_rng(12345)
    X = rng.uniform(low=-limit, high=limit, size=n)
    Y = rng.uniform(low=-limit, high=limit, size=n)
    Z = rng.uniform(low=-limit, high=limit, size=n)

    K = X * Y - Z*Z
    H = (X + Y) / 2

    kappa_1 = H + np.sqrt(H*H - K)
    kappa_2 = H - np.sqrt(H*H - K)

    bla = np.stack((kappa_1, kappa_2), axis=0)

    tt = np.corrcoef(bla)

    print(tt)