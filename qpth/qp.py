import os
import sys

import numpy as np
import numpy.random as npr
# np.set_printoptions(precision=5)

import cvxpy as cp

import torch
from torch.autograd import Function, Variable
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

from block import block

import time

from .util import bger, bdiag, expandParam, extract_nBatch
from .solvers.pdipm import batch as pdipm_b
from .solvers.pdipm import single as pdipm_s

class QPFunction(Function):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def forward(self, Q_, p_, G_, h_, A_, b_):
        """Solve a batch of QPs.

        This function solves a batch of QPs, each optimizing over
        `nz` variables and having `nineq` inequality constraints
        and `neq` equality constraints.
        The optimization problem for each instance in the batch
        (dropping indexing from the notation) is of the form

            \hat z =   argmin_z 1/2 z^T Q z + p^T z
                     subject to Gz <= h
                                Az  = b

        where Q \in S^{nz,nz},
              S^{nz,nz} is the set of all positive semi-definite matrices,
              p \in R^{nz}
              G \in R^{nineq,nz}
              h \in R^{nineq}
              A \in R^{neq,nz}
              b \in R^{neq}

        These parameters should all be passed to this function as
        Variable- or Parameter-wrapped Tensors.
        (See torch.autograd.Variable and torch.nn.parameter.Parameter)

        If you want to solve a batch of QPs where `nz`, `nineq` and `neq`
        are the same, but some of the contents differ across the
        minibatch, you can pass in tensors in the standard way
        where the first dimension indicates the batch example.
        This can be done with some or all of the coefficients.

        You do not need to add an extra dimension to coefficients
        that will not change across all of the minibatch examples.
        This function is able to infer such cases.

        If you don't want to use any equality or inequality constraints,
        you can set the appropriate values to:

            e = Variable(torch.Tensor())

        Parameters:
          Q:  A (nBatch, nz, nz) or (nz, nz) Tensor.
          p:  A (nBatch, nz) or (nz) Tensor.
          G:  A (nBatch, nineq, nz) or (nineq, nz) Tensor.
          h:  A (nBatch, nineq) or (nineq) Tensor.
          A:  A (nBatch, neq, nz) or (neq, nz) Tensor.
          b:  A (nBatch, neq) or (neq) Tensor.

        Returns: \hat z: a (nBatch, nz) Tensor.
        """
        start = time.time()
        nBatch = extract_nBatch(Q_, p_, G_, h_, A_, b_)
        Q, _ = expandParam(Q_, nBatch, 3)
        p, _ = expandParam(p_, nBatch, 2)
        G, _ = expandParam(G_, nBatch, 3)
        h, _ = expandParam(h_, nBatch, 2)
        A, _ = expandParam(A_, nBatch, 3)
        b, _ = expandParam(b_, nBatch, 2)

        _, nineq, nz = G.size()
        neq = A.size(1) if A.ndimension() > 0 else 0
        assert(neq > 0 or nineq > 0)
        self.neq, self.nineq, self.nz = neq, nineq, nz

        self.Q_LU, self.S_LU, self.R = pdipm_b.pre_factor_kkt(Q, G, A)

        zhats, self.nus, self.lams, self.slacks = pdipm_b.forward(
            Q, p, G, h, A, b, self.Q_LU, self.S_LU, self.R,
            self.verbose)

        self.save_for_backward(zhats, Q_, p_, G_, h_, A_, b_)
        # print('  + Forward pass took {:0.4f} seconds.'.format(time.time()-start))
        return zhats

    def backward(self, dl_dzhat):
        start = time.time()
        zhats, Q, p, G, h, A, b = self.saved_tensors
        nBatch = extract_nBatch(Q, p, G, h, A, b)
        Q, Q_e = expandParam(Q, nBatch, 3)
        p, p_e = expandParam(p, nBatch, 2)
        G, G_e = expandParam(G, nBatch, 3)
        h, h_e = expandParam(h, nBatch, 2)
        A, A_e = expandParam(A, nBatch, 3)
        b, b_e = expandParam(b, nBatch, 2)

        neq, nineq, nz = self.neq, self.nineq, self.nz

        d = self.lams/self.slacks
        pdipm_b.factor_kkt(self.S_LU, self.R, d)
        dx, _, dlam, dnu = pdipm_b.solve_kkt(
            self.Q_LU, d, G, A, self.S_LU,
            dl_dzhat, torch.zeros(nBatch, nineq).type_as(G),
            torch.zeros(nBatch, nineq).type_as(G),
            torch.zeros(nBatch, neq).type_as(G))

        dps = dx
        dGs = bger(dlam, zhats) + bger(self.lams, dx)
        if G_e:
            dGs = dGs.mean(0).squeeze(0)
        dhs = -dlam
        if h_e:
            dhs = dhs.mean(0).squeeze(0)
        if neq > 0:
            dAs = bger(dnu, zhats) + bger(self.nus, dx)
            dbs = -dnu
            if A_e:
                dAs = dAs.mean(0).squeeze(0)
            if b_e:
                dbs = dbs.mean(0).squeeze(0)
        else:
            dAs, dbs = None, None
        dQs = 0.5*(bger(dx, zhats) + bger(zhats, dx))
        if Q_e:
            dQs = dQs.mean(0).squeeze(0)

        # I = 1-torch.tril(torch.ones(nz,nz)).repeat(nBatch, 1, 1).type_as(G).byte()
        # dLs[I] = 0.0
        # import IPython, sys; IPython.embed(); sys.exit(-1)

        # if neq > 0:
        #     from block import block
        #     Q = Q[0]
        #     G = G[0]
        #     A = A[0]
        #     h = h[0]
        #     K1 = block(((Q, G.t(), A.t()),
        #                 (torch.diag(self.lams[0]).mm(G), torch.diag(G.mv(zhats[0])-h), 0),
        #                 (A, 0, 0))).t().cpu().numpy()
        #     rhs1 = torch.cat((-dl_dzhat[0],
        #                     torch.zeros(nineq+neq).type_as(G))).cpu().numpy()
        #     dx1, dlam1, dnu1 = np.split(np.linalg.solve(K1, rhs1), [nz, nz+nineq])
        #     import IPython, sys; IPython.embed(); sys.exit(-1)

        # K2 = block(((Q, 0, G.t(), A.t()),
        #             (0, torch.diag(d[0]), 'I', 0),
        #             (G, 'I', 0, 0),
        #             (A, 0, 0, 0))).cpu().numpy()
        # rhs2 = torch.cat((-dl_dzhat[0],
        #                  torch.zeros(2*nineq+neq).type_as(G))).cpu().numpy()
        # dx2, _, dlam2, dnu2 = np.split(np.linalg.solve(K2, rhs2), [nz, nz+nineq, nz+2*nineq])

        # K3 = block(((Q, G.t()),
        #             (torch.diag(self.lams[0]).mm(G), torch.diag(G.mv(zhats[0])-h)))
        # ).t().cpu().numpy()
        # rhs3 = torch.cat((-dl_dzhat[0],
        #                  torch.zeros(nineq).type_as(G))).cpu().numpy()
        # dx3, dlam3 = np.split(np.linalg.solve(K3, rhs3), [nz])

        # K4 = block(((Q, 0, G.t()),
        #             (0, torch.diag(d[0]), 'I'),
        #             (G, 'I', 0))).cpu().numpy()
        # rhs4 = torch.cat((-dl_dzhat[0],
        #                  torch.zeros(2*nineq).type_as(G))).cpu().numpy()
        # dx4, _, dlam4 = np.split(np.linalg.solve(K4, rhs4), [nz, nz+nineq])

        # dh_fd:  [ -1.879067e-04   2.413588e+00   1.059652e+00]
        # dh:  [-0.099737  2.444964  1.10988 ]
        # import IPython, sys; IPython.embed(); sys.exit(-1)

        # dQ = dQs.mean(0).squeeze()
        # dG, dh = [x.mean(0).squeeze() for x in [dGs, dhs]] \
        #          if nineq > 0 else [torch.Tensor().type_as(Q)]*2
        # dA, db = [x.mean(0).squeeze() for x in [dAs, dbs]] \
        #          if neq > 0 else [torch.Tensor().type_as(Q)]*2
        grads = (dQs, dps, dGs, dhs, dAs, dbs)
        # print('  + Backward pass took {:0.4f} seconds.'.format(time.time()-start))

        return grads

    def forward_serialized(self, input, Q, p, G, h, A, b):
        assert False, 'needs updating'
        start = time.time()
        nineq, nz = G.size()
        neq = A.size(0) if A.ndimension() > 0 else 0
        assert(neq > 0 or nineq > 0)
        assert(input.dim() == 2)
        nBatch, nz = input.size()
        self.neq, self.nineq, self.nz = neq, nineq, nz

        Q = torch.mm(L, L.t())+self.eps*torch.eye(nz).type_as(L) # Could be cached.
        self.Q = Q
        b = torch.mv(A, z0) if self.neq > 0 else None
        h = torch.mv(G,z0)+s0
        L_Q, L_S, R = qp.pre_factor_kkt(Q, G, A, nineq, neq)

        zhats = torch.zeros(nBatch, nz).type_as(L)
        nus = torch.zeros(nBatch, neq).type_as(L) if neq > 0 else None
        lams = torch.zeros(nBatch, nineq).type_as(L) if nineq > 0 else None
        for i in range(nBatch):
            zhat_i, nu_i, lam_i = qp.forward_single(input[i], Q, G, A, b, h, L_Q, L_S, R)
            zhats[i] = zhat_i
            if neq > 0:
                nus[i] = nu_i
            if nineq > 0:
                lams[i] = lam_i

        self.nus = nus
        self.lams = lams

        self.save_for_backward(input, zhats, L, G, A, z0, s0)
        # print('  + Forward pass took {:0.4f} seconds.'.format(time.time()-start))
        return zhats

# class QPLayer(Module):
#     def __init__(self, Q, G, h, A, b):
#         super(QPLayer, self).__init__()
#         self.Q, self.G, self.h, self.A, self.b = \
#             [Parameter(x) for x in [Q, G, h, A, b]]

#     def forward(self, inputs):
#         return QPFunction()(inputs, self.Q, self.p, self.G, self.h, self.A, self.b)
