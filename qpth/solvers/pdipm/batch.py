import torch
import sys
from qpth.util import get_sizes

def forward(inputs, Q, G, h, A, b, Q_LU, S_LU, R):
    """
    b = A z_0
    h = G z_0 + s_0
    Q_LU, S_LU, R = pre_factor_kkt(Q, G, A)
    """
    nineq, nz, neq, nBatch = get_sizes(G, A, inputs)

    # Find initial values
    d = torch.ones(nBatch,nineq).type_as(Q)
    factor_kkt(S_LU, R, d)
    x, s, z, y = solve_kkt(
        Q_LU, d, G, A, S_LU,
        inputs, torch.zeros(nBatch, nineq).type_as(Q),
        -h.unsqueeze(0).expand(nBatch, nineq),
        -b.unsqueeze(0).expand(nBatch, neq) if neq > 0 else None)
    # D = torch.eye(nineq).repeat(nBatch, 1, 1).type_as(Q)
    # x1, s1, z1, y1 = factor_solve_kkt(
    #     Q, D, G, A,
    #     inputs, torch.zeros(nBatch, nineq).type_as(Q),
    #     -h.repeat(nBatch, 1),
    #     nb.repeat(nBatch, 1) if b is not None else None)
    # U_Q, U_S, R = pre_factor_kkt(Q, G, A)
    # factor_kkt(U_S, R, d[0])
    # x2, s2, z2, y2 = solve_kkt(
    #     U_Q, d[0], G, A, U_S,
    #     inputs[0], torch.zeros(nineq).type_as(Q), -h, nb)
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
        rx = (torch.mm(y, A) if neq > 0 else 0.) + \
             torch.mm(z, G) + torch.mm(x, Q.t()) + inputs
        rs = z
        rz = torch.mm(x, G.t()) + s - h.repeat(nBatch, 1)
        ry = torch.mm(x, A.t()) - b.repeat(nBatch, 1) if neq > 0 else 0.0
        mu = (s*z).sum(1).squeeze()/nineq
        z_resid = torch.norm(rz, 2, 1).squeeze()
        y_resid = torch.norm(ry, 2, 1).squeeze() if neq > 0 else 0
        pri_resid = y_resid + z_resid
        dual_resid = torch.norm(rx, 2, 1).squeeze()
        resids = pri_resid + dual_resid + nineq*mu
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
        d = z/s
        if nNotImproved == 3 or best['resids'].max() < 1e-12:
            return best['x'], best['y'], best['z'], best['s']

        # L_Q, L_S, R_ = pre_factor_kkt(Q, G, A)
        # factor_kkt(L_S, R_, d[0])
        # dx_cor, ds_cor, dz_cor, dy_cor = solve_kkt(
        #     L_Q, d[0], G, A, L_S, rx[0], rs[0], rz[0], ry[0])
        factor_kkt(S_LU, R, d)
        dx_aff, ds_aff, dz_aff, dy_aff = solve_kkt(
            Q_LU, d, G, A, S_LU, rx, rs, rz, ry)

        # D = diaged(d)
        # dx_aff1, ds_aff1, dz_aff1, dy_aff1 = factor_solve_kkt(
        #     Q, D, G, A, rx, rs, rz, ry)
        # dx_aff2, ds_aff2, dz_aff2, dy_aff2 = factor_solve_kkt(
        #     Q, D[0], G, A, rx[0], rs[0], 0, ry[0])

        # compute centering directions
        # alpha0 = min(min(get_step(z[0],dz_aff[0]), get_step(s[0], ds_aff[0])), 1.0)
        alpha = torch.min(torch.min(get_step(z,dz_aff),
                                    get_step(s, ds_aff)),
                          torch.ones(nBatch).type_as(Q))
        alpha_nineq = alpha.repeat(nineq, 1).t()
        # alpha_nz = alpha.repeat(nz, 1).t()
        # sig0 = (torch.dot(s[0] + alpha[0]*ds_aff[0],
                          # z[0] + alpha[0]*dz_aff[0])/(torch.dot(s[0],z[0])))**3
        t1 = s + alpha_nineq*ds_aff
        t2 = z + alpha_nineq*dz_aff
        t3 = torch.sum(t1*t2, 1).squeeze()
        t4 = torch.sum(s*z, 1).squeeze()
        sig = (t3/t4)**3
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
        rs = ((-mu*sig).repeat(nineq,1).t() + ds_aff*dz_aff)/s
        rz = torch.zeros(nBatch, nineq).type_as(Q)
        ry = torch.zeros(nBatch, neq).type_as(Q)
        # dx_cor1, ds_cor1, dz_cor1, dy_cor1 = factor_solve_kkt(
        #     Q, D, G, A, rx, rs, rz, ry)
        dx_cor, ds_cor, dz_cor, dy_cor= solve_kkt(
            Q_LU, d, G, A, S_LU, rx, rs, rz, ry)

        dx = dx_aff + dx_cor
        ds = ds_aff + ds_cor
        dz = dz_aff + dz_cor
        dy = dy_aff + dy_cor if neq > 0 else None
        import qpth.solvers.pdipm.single as pdipm_s
        alpha0 = min(1.0, 0.999*min(pdipm_s.get_step(s[0],ds[0]), pdipm_s.get_step(z[0],dz[0])))
        alpha = torch.min(0.999*torch.min(get_step(z, dz),
                                          get_step(s, ds)),
                          torch.ones(nBatch).type_as(Q))
        assert(alpha0 - alpha[0] <= 1e-10) # TODO: Remove

        alpha_nineq = alpha.repeat(nineq, 1).t()
        alpha_neq = alpha.repeat(neq, 1).t() if neq > 0 else None
        alpha_nz = alpha.repeat(nz, 1).t()
        dx_norm = torch.norm(dx, 2, 1).squeeze()
        dz_norm = torch.norm(dz, 2, 1).squeeze()
        # if TODO ->np.any(np.isnan(dx_norm)) or \
        #    torch.sum(dx_norm > 1e5) > 0 or \
        #    torch.sum(dz_norm > 1e5):
        #     # Overflow, return early
        #     return x, y, z

        x += alpha_nz*dx
        s += alpha_nineq*ds
        z += alpha_nineq*dz
        y = y + alpha_neq*dy if neq > 0 else None

    return best['x'], best['y'], best['z'], best['s']

def get_step(v,dv):
    nBatch = v.size(0)
    a = -v/dv
    a[dv >= 1e-12] = max(1.0, a.max())
    return a.min(1)[0].squeeze()

def factor_solve_kkt(Q, D, G, A, rx, rs, rz, ry):
    nineq, nz, neq, nBatch = get_sizes(G, A, rx)

    if neq > 0:
        # import IPython, sys; IPython.embed(); sys.exit(-1)
        # H_ = torch.cat([torch.cat([Q, torch.zeros(nz,nineq).type_as(Q)], 1),
        #                 torch.cat([torch.zeros(nineq, nz).type_as(Q), D], 1)], 0)
        # A_ = torch.cat([torch.cat([G, torch.eye(nineq).type_as(Q)], 1),
        #                 torch.cat([A, torch.zeros(neq, nineq).type_as(Q)], 1)], 0)
        # g_ = torch.cat([rx, rs], 0)
        # h_ = torch.cat([rz, ry], 0)

        H_ = torch.zeros(nBatch, nz+nineq, nz+nineq).type_as(Q)
        H_[:,:nz,:nz] = Q.repeat(nBatch, 1, 1)
        H_[:,-nineq:,-nineq:] = D

        from block import block
        A_ = block(((G, 'I'),
                    (A, torch.zeros(neq, nineq).type_as(Q))))

        g_ = torch.cat([rx, rs], 1)
        h_ = torch.cat([rz, ry], 1)
    else:
        H_ = torch.zeros(nBatch, nz+nineq, nz+nineq).type_as(Q)
        H_[:,:nz,:nz] = Q.repeat(nBatch, 1, 1)
        H_[:,-nineq:,-nineq:] = D
        A_ = torch.cat([G, torch.eye(nineq).type_as(Q)], 1)
        g_ = torch.cat([rx, rs], 1)
        h_ = rz

    H_LU = H_.btrf()

    A = A_.repeat(nBatch, 1, 1)
    invH_A_= A.transpose(1, 2).btrs(H_LU)
    invH_g_ = g_.btrs(H_LU)

    S_ = torch.bmm(A, invH_A_)
    S_LU = S_.btrf()
    t_ = torch.mm(invH_g_, A_.t()) - h_
    w_ = -t_.btrs(S_LU)
    t_ = -g_-w_.mm(A_)
    v_ = t_.btrs(H_LU)

    dx = v_[:,:nz]
    ds = v_[:,nz:]
    dz = w_[:,:nineq]
    dy = w_[:,nineq:] if neq > 0 else None

    return dx, ds, dz, dy

def solve_kkt(Q_LU, d, G, A, S_LU, rx, rs, rz, ry, dbg=False):
    """ Solve KKT equations for the affine step"""
    nineq, nz, neq, nBatch = get_sizes(G, A, rx)

    invQ_rx = rx.t().unsqueeze(0).btrs(Q_LU.unsqueeze(0)).squeeze(0).t()
    # if rs.norm()+rz.norm()+ry.norm() == 0:
        # import IPython, sys; IPython.embed(); sys.exit(-1)
    if neq > 0:
        h = torch.cat((invQ_rx.mm(A.t()) - ry, invQ_rx.mm(G.t()) + rs/d - rz), 1)
    else:
        h = invQ_rx.mm(G.t()) + rs/d - rz

    w = -(h.btrs(S_LU))

    g1 = -rx - w[:,neq:].mm(G)
    if neq > 0:
        g1 -= w[:,:neq].mm(A)
    g2 = -rs - w[:,neq:]

    dx = g1.t().unsqueeze(0).btrs(Q_LU.unsqueeze(0)).squeeze(0).t()
    ds = g2/d
    dz = w[:,neq:]
    dy = w[:,:neq] if neq > 0 else None

    # if np.all(np.array([x.norm() for x in [rx, rs, rz, ry]]) != 0):
    if dbg:
        import IPython, sys; IPython.embed(); sys.exit(-1)

    return dx, ds, dz, dy

def pre_factor_kkt(Q, G, A, nBatch):
    """ Perform all one-time factorizations and cache relevant matrix products"""
    nineq, nz, neq, _ = get_sizes(G, A)

    Q_LU = Q.unsqueeze(0).btrf()

    # S = [ A Q^{-1} A^T        A Q^{-1} G^T          ]
    #     [ G Q^{-1} A^T        G Q^{-1} G^T + D^{-1} ]
    #
    # We compute a partial LU decomposition of S matrix
    # that can be completed once D^{-1} is known.
    # This is done for a general matrix by decomposing
    # S using the Schur complement and then LU-factorizing
    # the matrices in the middle:
    #
    #   [ A B ] = [ I            0 ] [ A     0              ] [ I    A^{-1} B ]
    #   [ C D ]   [ C A^{-1}     I ] [ 0     D - C A^{-1} B ] [ 0    I        ]

    G_invQ_GT = torch.mm(G, G.t().unsqueeze(0).btrs(Q_LU).squeeze(0))
    Q_LU = Q_LU.squeeze()
    R = G_invQ_GT
    if neq > 0:
        invQ_AT = A.t().unsqueeze(0).btrs(Q_LU.unsqueeze(0)).squeeze(0)
        A_invQ_AT = torch.mm(A, invQ_AT)
        G_invQ_AT = torch.mm(G, invQ_AT)

        LU_A_invQ_AT = A_invQ_AT.unsqueeze(0).btrf().squeeze(0)
        L_A_invQ_AT = torch.tril(LU_A_invQ_AT)
        L_A_invQ_AT[torch.eye(neq).type_as(Q).byte()] = 1.0
        U_A_invQ_AT = torch.triu(LU_A_invQ_AT)

        S_LU_11 = LU_A_invQ_AT
        S_LU_21 = G_invQ_AT.mm(L_A_invQ_AT.unsqueeze(0).btrs(
            LU_A_invQ_AT.unsqueeze(0)).squeeze(0))
        T = G_invQ_AT.t().unsqueeze(0).btrs(LU_A_invQ_AT.unsqueeze(0)).squeeze(0)
        S_LU_12 = U_A_invQ_AT.mm(T)
        S_LU = torch.cat((torch.cat((S_LU_11, S_LU_12), 1),
                          torch.cat((S_LU_21, torch.zeros(nineq,nineq).type_as(Q)), 1)))
        S_LU = S_LU.repeat(nBatch, 1, 1)
        R -= G_invQ_AT.mm(T)
    else:
        S_LU = torch.zeros(nBatch, nineq, nineq).type_as(Q)

    R = R.repeat(nBatch, 1, 1)

    return Q_LU, S_LU, R

factor_kkt_eye = None

def factor_kkt(S_LU, R, d):
    """ Factor the U22 block that we can only do after we know D. """
    nBatch, nineq = d.size()
    # TODO: There's probably a better way to add a batched diagonal.
    global factor_kkt_eye
    if factor_kkt_eye is None or factor_kkt_eye.size() != d.size():
        # print('Updating batchedEye size.')
        factor_kkt_eye = torch.eye(nineq).repeat(nBatch,1,1).type_as(R).byte()
    T = R.clone()
    T[factor_kkt_eye] += (1./d).squeeze()
    S_LU[:,-nineq:,-nineq:] = T.btrf()
