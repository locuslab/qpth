import torch
from qpth.util import get_sizes

from block import block


def forward(Q, p, G, h, A, b, Q_LU, S_LU, R, verbose=False):
    """
    Q_LU, S_LU, R = pre_factor_kkt(Q, G, A)
    """
    nineq, nz, neq, nBatch = get_sizes(G, A)

    # Find initial values
    d = torch.ones(nBatch, nineq).type_as(Q)
    factor_kkt(S_LU, R, d)
    x, s, z, y = solve_kkt(
        Q_LU, d, G, A, S_LU,
        p, torch.zeros(nBatch, nineq).type_as(Q),
        -h, -b if neq > 0 else None)
    # D = torch.eye(nineq).repeat(nBatch, 1, 1).type_as(Q)
    # x1, s1, z1, y1 = factor_solve_kkt(
    #     Q, D, G, A,
    #     p, torch.zeros(nBatch, nineq).type_as(Q),
    #     -h.repeat(nBatch, 1),
    #     nb.repeat(nBatch, 1) if b is not None else None)
    # U_Q, U_S, R = pre_factor_kkt(Q, G, A)
    # factor_kkt(U_S, R, d[0])
    # x2, s2, z2, y2 = solve_kkt(
    #     U_Q, d[0], G, A, U_S,
    #     p[0], torch.zeros(nineq).type_as(Q), -h, nb)
    # import IPython, sys; IPython.embed(); sys.exit(-1)

    M = torch.min(s, 1)[0].repeat(1, nineq)
    I = M < 0
    s[I] -= M[I] - 1

    M = torch.min(z, 1)[0].repeat(1, nineq)
    I = M < 0
    z[I] -= M[I] - 1

    best = {'resids': None, 'x': None, 'z': None, 's': None, 'y': None}
    nNotImproved = 0

    for i in range(20):
        # affine scaling direction
        rx = (torch.bmm(y.unsqueeze(1), A).squeeze(1) if neq > 0 else 0.) + \
            torch.bmm(z.unsqueeze(1), G).squeeze(1) + \
            torch.bmm(x.unsqueeze(1), Q.transpose(1, 2)).squeeze(1) + \
            p
        rs = z
        rz = torch.bmm(x.unsqueeze(1), G.transpose(1, 2)).squeeze(1) + s - h
        ry = torch.bmm(x.unsqueeze(1), A.transpose(
            1, 2)).squeeze(1) - b if neq > 0 else 0.0
        mu = torch.abs((s * z).sum(1).squeeze() / nineq)
        z_resid = torch.norm(rz, 2, 1).squeeze()
        y_resid = torch.norm(ry, 2, 1).squeeze() if neq > 0 else 0
        pri_resid = y_resid + z_resid
        dual_resid = torch.norm(rx, 2, 1).squeeze()
        resids = pri_resid + dual_resid + nineq * mu

        d = z / s
        try:
            factor_kkt(S_LU, R, d)
        except:
            return best['x'], best['y'], best['z'], best['s']

        if verbose:
            print('iter: {}, pri_resid: {:.5e}, dual_resid: {:.5e}, mu: {:.5e}'.format(
                i, pri_resid[0], dual_resid[0], mu[0]))
        if best['resids'] is None:
            best['resids'] = resids
            best['x'] = x.clone()
            best['z'] = z.clone()
            best['s'] = s.clone()
            best['y'] = y.clone() if y is not None else None
            nNotImproved = 0
        else:
            I = resids < best['resids']
            if I.sum() > 0:
                nNotImproved = 0
            else:
                nNotImproved += 1
            I_nz = I.repeat(nz, 1).t()
            I_nineq = I.repeat(nineq, 1).t()
            best['resids'][I] = resids[I]
            best['x'][I_nz] = x[I_nz]
            best['z'][I_nineq] = z[I_nineq]
            best['s'][I_nineq] = s[I_nineq]
            if neq > 0:
                I_neq = I.repeat(neq, 1).t()
                best['y'][I_neq] = y[I_neq]
        if nNotImproved == 3 or best['resids'].max() < 1e-12:
            return best['x'], best['y'], best['z'], best['s']

        # L_Q, L_S, R_ = pre_factor_kkt(Q, G, A)
        # factor_kkt(L_S, R_, d[0])
        # dx_cor, ds_cor, dz_cor, dy_cor = solve_kkt(
        #     L_Q, d[0], G, A, L_S, rx[0], rs[0], rz[0], ry[0])
        # TODO: Move factorization back here.
        dx_aff, ds_aff, dz_aff, dy_aff = solve_kkt(
            Q_LU, d, G, A, S_LU, rx, rs, rz, ry)

        # D = diaged(d)
        # dx_aff1, ds_aff1, dz_aff1, dy_aff1 = factor_solve_kkt(
        #     Q, D, G, A, rx, rs, rz, ry)
        # dx_aff2, ds_aff2, dz_aff2, dy_aff2 = factor_solve_kkt(
        #     Q, D[0], G, A, rx[0], rs[0], 0, ry[0])

        # compute centering directions
        # alpha0 = min(min(get_step(z[0],dz_aff[0]), get_step(s[0], ds_aff[0])), 1.0)
        alpha = torch.min(torch.min(get_step(z, dz_aff),
                                    get_step(s, ds_aff)),
                          torch.ones(nBatch).type_as(Q))
        alpha_nineq = alpha.repeat(nineq, 1).t()
        # alpha_nz = alpha.repeat(nz, 1).t()
        # sig0 = (torch.dot(s[0] + alpha[0]*ds_aff[0],
        # z[0] + alpha[0]*dz_aff[0])/(torch.dot(s[0],z[0])))**3
        t1 = s + alpha_nineq * ds_aff
        t2 = z + alpha_nineq * dz_aff
        t3 = torch.sum(t1 * t2, 1).squeeze()
        t4 = torch.sum(s * z, 1).squeeze()
        sig = (t3 / t4)**3
        # dx_cor, ds_cor, dz_cor, dy_cor = solve_kkt(
        #     U_Q, d, G, A, U_S, torch.zeros(nz).type_as(Q),
        #     (-mu*sig*torch.ones(nineq).type_as(Q) + ds_aff*dz_aff)/s,
        #     torch.zeros(nineq).type_as(Q), torch.zeros(neq).type_as(Q), neq, nz)
        # D = diaged(d)
        # dx_cor0, ds_cor0, dz_cor0, dy_cor0 = factor_solve_kkt(Q, D[0], G, A,
        #     torch.zeros(nz).type_as(Q),
        #     (-mu[0]*sig[0]*torch.ones(nineq).type_as(Q)+ds_aff[0]*dz_aff[0])/s[0],
        #     torch.zeros(nineq).type_as(Q), torch.zeros(neq).type_as(Q))
        rx = torch.zeros(nBatch, nz).type_as(Q)
        rs = ((-mu * sig).repeat(nineq, 1).t() + ds_aff * dz_aff) / s
        rz = torch.zeros(nBatch, nineq).type_as(Q)
        ry = torch.zeros(nBatch, neq).type_as(Q)
        # dx_cor1, ds_cor1, dz_cor1, dy_cor1 = factor_solve_kkt(
        #     Q, D, G, A, rx, rs, rz, ry)
        dx_cor, ds_cor, dz_cor, dy_cor = solve_kkt(
            Q_LU, d, G, A, S_LU, rx, rs, rz, ry)

        dx = dx_aff + dx_cor
        ds = ds_aff + ds_cor
        dz = dz_aff + dz_cor
        dy = dy_aff + dy_cor if neq > 0 else None
        # import qpth.solvers.pdipm.single as pdipm_s
        # alpha0 = min(1.0, 0.999*min(pdipm_s.get_step(s[0],ds[0]), pdipm_s.get_step(z[0],dz[0])))
        alpha = torch.min(0.999 * torch.min(get_step(z, dz),
                                            get_step(s, ds)),
                          torch.ones(nBatch).type_as(Q))
        # assert(np.isnan(alpha0) or np.isinf(alpha0) or alpha0 - alpha[0] <=
        # 1e-5) # TODO: Remove

        alpha_nineq = alpha.repeat(nineq, 1).t()
        alpha_neq = alpha.repeat(neq, 1).t() if neq > 0 else None
        alpha_nz = alpha.repeat(nz, 1).t()
        # dx_norm = torch.norm(dx, 2, 1).squeeze()
        # dz_norm = torch.norm(dz, 2, 1).squeeze()
        # if TODO ->np.any(np.isnan(dx_norm)) or \
        #    torch.sum(dx_norm > 1e5) > 0 or \
        #    torch.sum(dz_norm > 1e5):
        #     # Overflow, return early
        #     return x, y, z

        x += alpha_nz * dx
        s += alpha_nineq * ds
        z += alpha_nineq * dz
        y = y + alpha_neq * dy if neq > 0 else None

    return best['x'], best['y'], best['z'], best['s']


def get_step(v, dv):
    # nBatch = v.size(0)
    a = -v / dv
    a[dv >= 1e-12] = max(1.0, a.max())
    return a.min(1)[0].squeeze()


def factor_solve_kkt(Q, D, G, A, rx, rs, rz, ry):
    nineq, nz, neq, nBatch = get_sizes(G, A)

    if neq > 0:
        # import IPython, sys; IPython.embed(); sys.exit(-1)
        # H_ = torch.cat([torch.cat([Q, torch.zeros(nz,nineq).type_as(Q)], 1),
        #                 torch.cat([torch.zeros(nineq, nz).type_as(Q), D], 1)], 0)
        # A_ = torch.cat([torch.cat([G, torch.eye(nineq).type_as(Q)], 1),
        #                 torch.cat([A, torch.zeros(neq, nineq).type_as(Q)], 1)], 0)
        # g_ = torch.cat([rx, rs], 0)
        # h_ = torch.cat([rz, ry], 0)

        H_ = torch.zeros(nBatch, nz + nineq, nz + nineq).type_as(Q)
        H_[:, :nz, :nz] = Q.repeat(nBatch, 1, 1)
        H_[:, -nineq:, -nineq:] = D

        A_ = block(((G, 'I'),
                    (A, torch.zeros(neq, nineq).type_as(Q))))

        g_ = torch.cat([rx, rs], 1)
        h_ = torch.cat([rz, ry], 1)
    else:
        H_ = torch.zeros(nBatch, nz + nineq, nz + nineq).type_as(Q)
        H_[:, :nz, :nz] = Q.repeat(nBatch, 1, 1)
        H_[:, -nineq:, -nineq:] = D
        A_ = torch.cat([G, torch.eye(nineq).type_as(Q)], 1)
        g_ = torch.cat([rx, rs], 1)
        h_ = rz

    H_LU = H_.btrifact()

    A = A_.repeat(nBatch, 1, 1)
    invH_A_ = A.transpose(1, 2).btrisolve(*H_LU)
    invH_g_ = g_.btrisolve(*H_LU)

    S_ = torch.bmm(A, invH_A_)
    S_LU = S_.btrifact()
    t_ = torch.mm(invH_g_, A_.t()) - h_
    w_ = -t_.btrisolve(*S_LU)
    t_ = -g_ - w_.mm(A_)
    v_ = t_.btrisolve(*H_LU)

    dx = v_[:, :nz]
    ds = v_[:, nz:]
    dz = w_[:, :nineq]
    dy = w_[:, nineq:] if neq > 0 else None

    return dx, ds, dz, dy


def solve_kkt(Q_LU, d, G, A, S_LU, rx, rs, rz, ry):
    """ Solve KKT equations for the affine step"""
    nineq, nz, neq, nBatch = get_sizes(G, A)

    invQ_rx = rx.btrisolve(*Q_LU)
    if neq > 0:
        h = torch.cat((invQ_rx.unsqueeze(1).bmm(A.transpose(1, 2)) - ry,
                       invQ_rx.unsqueeze(1).bmm(G.transpose(1, 2)) + rs / d - rz), 2).squeeze(1)
    else:
        h = (invQ_rx.unsqueeze(1).bmm(G.transpose(1, 2)) + rs / d - rz).squeeze(1)

    w = -(h.btrisolve(*S_LU))

    g1 = -rx - w[:, neq:].unsqueeze(1).bmm(G)
    if neq > 0:
        g1 -= w[:, :neq].unsqueeze(1).bmm(A)
    g2 = -rs - w[:, neq:]

    dx = g1.btrisolve(*Q_LU)
    ds = g2 / d
    dz = w[:, neq:]
    dy = w[:, :neq] if neq > 0 else None

    return dx, ds, dz, dy


def pre_factor_kkt(Q, G, A):
    """ Perform all one-time factorizations and cache relevant matrix products"""
    nineq, nz, neq, nBatch = get_sizes(G, A)

    Q_LU = Q.btrifact()

    # S = [ A Q^{-1} A^T        A Q^{-1} G^T          ]
    #     [ G Q^{-1} A^T        G Q^{-1} G^T + D^{-1} ]
    #
    # We compute a partial LU decomposition of the S matrix
    # that can be completed once D^{-1} is known.
    # See the 'Block LU factorization' part of our website
    # for more details.

    G_invQ_GT = torch.bmm(G, G.transpose(1, 2).btrisolve(*Q_LU))
    R = G_invQ_GT.clone()
    S_LU_pivots = torch.IntTensor(range(1, 1 + neq + nineq)).unsqueeze(0) \
        .repeat(nBatch, 1).type_as(Q).int()
    if neq > 0:
        invQ_AT = A.transpose(1, 2).btrisolve(*Q_LU)
        A_invQ_AT = torch.bmm(A, invQ_AT)
        G_invQ_AT = torch.bmm(G, invQ_AT)

        LU_A_invQ_AT = A_invQ_AT.btrifact()
        P_A_invQ_AT, L_A_invQ_AT, U_A_invQ_AT = torch.btriunpack(*LU_A_invQ_AT)
        P_A_invQ_AT = P_A_invQ_AT.type_as(A_invQ_AT)

        S_LU_11 = LU_A_invQ_AT[0]
        U_A_invQ_AT_inv = (P_A_invQ_AT.bmm(L_A_invQ_AT)
                           ).btrisolve(*LU_A_invQ_AT)
        S_LU_21 = G_invQ_AT.bmm(U_A_invQ_AT_inv)
        T = G_invQ_AT.transpose(1, 2).btrisolve(*LU_A_invQ_AT)
        S_LU_12 = U_A_invQ_AT.bmm(T)
        S_LU_22 = torch.zeros(nBatch, nineq, nineq).type_as(Q)
        S_LU_data = torch.cat((torch.cat((S_LU_11, S_LU_12), 2),
                               torch.cat((S_LU_21, S_LU_22), 2)),
                              1)
        S_LU_pivots[:, :neq] = LU_A_invQ_AT[1]

        R -= G_invQ_AT.bmm(T)
    else:
        S_LU_data = torch.zeros(nBatch, nineq, nineq).type_as(Q)

    S_LU = [S_LU_data, S_LU_pivots]
    return Q_LU, S_LU, R


factor_kkt_eye = None


def factor_kkt(S_LU, R, d):
    """ Factor the U22 block that we can only do after we know D. """
    nBatch, nineq = d.size()
    neq = S_LU[1].size(1) - nineq
    # TODO: There's probably a better way to add a batched diagonal.
    global factor_kkt_eye
    if factor_kkt_eye is None or factor_kkt_eye.size() != d.size():
        # print('Updating batchedEye size.')
        factor_kkt_eye = torch.eye(nineq).repeat(
            nBatch, 1, 1).type_as(R).byte()
    T = R.clone()
    T[factor_kkt_eye] += (1. / d).squeeze()
    T_LU = T.btrifact()
    oldPivotsPacked = S_LU[1][:, -nineq:] - neq
    oldPivots, _, _ = torch.btriunpack(
        T_LU[0], oldPivotsPacked, unpack_data=False)
    newPivotsPacked = T_LU[1]
    newPivots, _, _ = torch.btriunpack(
        T_LU[0], newPivotsPacked, unpack_data=False)

    # Re-pivot the S_LU_21 block.
    if neq > 0:
        S_LU_21 = S_LU[0][:, -nineq:, :neq]
        S_LU[0][:, -nineq:,
                :neq] = newPivots.transpose(1, 2).bmm(oldPivots.bmm(S_LU_21))

    # Add the new S_LU_22 block.
    S_LU[0][:, -nineq:, -nineq:] = T_LU[0]
    S_LU[1][:, -nineq:] = newPivotsPacked + neq
