import numpy as np

def backward_single_np_solver(self, S, zhat_i, nu_i, lam_i, dl_dzhat_i,
                              L, G, A, z0, s0):
    nz, neq, nineq = zhat_i.shape[0], A.shape[0], G.shape[0]

    y = np.dot(G, zhat_i) - np.dot(G, z0) - s0
    Y = np.diag(y)
    Lam = np.diag(lam_i)

    I = np.eye(nz + neq + nineq)
    Bz = S.solve([I[:nz], I[nz:nz + neq], I[nz + neq:]], d='z')
    Bz1, Bz2, Bz3 = np.split(Bz, [nz, nz + neq], axis=1)

    # TODO: These could potentially be obtained by the system
    # solve since some of these submatrices seem symmetric.
    dp = -np.array(np.dot(dl_dzhat_i, Bz1)).ravel()
    ds0 = np.array(np.dot(dl_dzhat_i, np.dot(Bz3, Lam))).ravel()
    dz0 = np.array(np.dot(dl_dzhat_i, np.dot(Bz2, A) +
                          np.dot(Bz3, np.dot(Lam, G)))).ravel()

    if neq > 0:
        dA_1 = np.dot(Bz1, np.kron(nu_i, np.eye(nz)))
        dA_2 = np.dot(Bz2, np.kron(np.eye(neq), zhat_i))
        dA = np.array(np.dot(dl_dzhat_i, -dA_1 - dA_2)).reshape((neq, nz))
    else:
        dA = None

    if nineq > 0:
        dG = -np.outer(lam_i, dl_dzhat_i.dot(Bz1)) - \
            np.outer(dl_dzhat_i.dot(Bz3).dot(Lam), zhat_i - z0)
    else:
        dG = None

    dL = np.tril(-np.outer(dl_dzhat_i.dot(Bz1), L.T.dot(zhat_i)) -
                 np.outer(zhat_i, dl_dzhat_i.dot(Bz1)).dot(L))

    return dp, dL, dG, dA, dz0, ds0
