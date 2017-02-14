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

from qpth.util import bger, bdiag
from qpth import solvers
import qpth.solvers.pdipm.single as pdipm_s
import qpth.solvers.pdipm.batch as pdipm_b

class QPFunction(Function):
    def forward(self, inputs, Q, G, h, A, b):
        start = time.time()
        nineq, nz = G.size()
        neq = A.size(0) if A.ndimension() > 0 else 0
        assert(neq > 0 or nineq > 0)
        assert(inputs.dim() == 2)
        nBatch, nz = inputs.size()
        self.neq, self.nineq, self.nz = neq, nineq, nz

        self.Q_LU, self.S_LU, self.R = pdipm_b.pre_factor_kkt(Q, G, A, nBatch)

        zhats, self.nus, self.lams, self.slacks = pdipm_b.forward(
            inputs, Q, G, h, A, b, self.Q_LU, self.S_LU, self.R)

        self.save_for_backward(inputs, zhats, Q, G, h, A, b)
        print('  + Forward pass took {:0.4f} seconds.'.format(time.time()-start))
        return zhats

    def backward(self, dl_dzhat):
        start = time.time()
        inputs, zhats, Q, G, h, A, b = self.saved_tensors

        nBatch = inputs.size(0)
        neq, nineq, nz = self.neq, self.nineq, self.nz

        d = self.lams/self.slacks
        pdipm_b.factor_kkt(self.S_LU, self.R, d)
        dx, _, dlam, dnu = pdipm_b.solve_kkt(
            self.Q_LU, d, G, A, self.S_LU,
            dl_dzhat, torch.zeros(nBatch, nineq).type_as(G),
            torch.zeros(nBatch, nineq).type_as(G),
            torch.zeros(nBatch, neq).type_as(G))

        dps = dx
        dbs = -dnu if neq > 0 else None
        dhs = -self.lams*dlam
        dAs = (bger(dnu, zhats) + bger(self.nus, dx)) if neq > 0 else None
        dGs = torch.bmm(bdiag(self.lams), bger(dlam, zhats)) + bger(self.lams, dx)
        dQs = 0.5*(bger(dx, zhats) + bger(zhats, dx))

        # I = 1-torch.tril(torch.ones(nz,nz)).repeat(nBatch, 1, 1).type_as(G).byte()
        # dLs[I] = 0.0
        # ds0s = dhs
        # dz0s = dhs.mm(G)+(dbs.mm(A) if neq > 0 else 0)
        # import IPython, sys; IPython.embed(); sys.exit(-1)

        # from block import block
        # h = G.mv(z0) + s0
        # b = A.mv(z0) if neq > 0 else None
        # K1 = block(((self.Q, G.t(), A.t()),
        #             (torch.diag(self.lams[0]).mm(G), torch.diag(G.mv(zhats[0])-h), 0),
        #             (A, 0, 0))).t().cpu().numpy()
        # rhs1 = torch.cat((-dl_dzhat[0],
        #                  torch.zeros(nineq+neq).type_as(G))).cpu().numpy()
        # dx1, dlam1, dnu1 = np.split(np.linalg.solve(K1, rhs1), [nz, nz+nineq])

        # K2 = block(((self.Q, 0, G.t(), A.t()),
        #             (0, torch.diag(d[0]), 'I', 0),
        #             (G, 'I', 0, 0),
        #             (A, 0, 0, 0))).cpu().numpy()
        # rhs2 = torch.cat((-dl_dzhat[0],
        #                  torch.zeros(2*nineq+neq).type_as(G))).cpu().numpy()
        # dx2, _, dlam2, dnu2 = np.split(np.linalg.solve(K2, rhs2), [nz, nz+nineq, nz+2*nineq])

        # import IPython, sys; IPython.embed(); sys.exit(-1)
        # print(dhs, dbs, dhs.mm(G), dbs.mm(A))
        # [-0.94  1.33  2.5  -1.36 -1.04 -0.15 -0.63  0.13 -0.28  0.27]
        # import IPython, sys; IPython.embed(); sys.exit(-1)

        dQ = dQs.mean(0).squeeze()
        dG, dh = [x.mean(0).squeeze() for x in [dGs, dhs]] \
                 if nineq > 0 else [torch.Tensor().type_as(Q)]*2
        dA, db = [x.mean(0).squeeze() for x in [dAs, dbs]] \
                 if neq > 0 else [torch.Tensor().type_as(Q)]*2
        grads = (dps, dQ, dG, dh, dA, db)
        print('  + Backward pass took {:0.4f} seconds.'.format(time.time()-start))

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
        print('  + Forward pass took {:0.4f} seconds.'.format(time.time()-start))
        return zhats


class QPLayer(Module):
    def __init__(self, Q, G, h, A, b):
        super(QPLayer, self).__init__()
        self.Q, self.G, self.h, self.A, self.b = \
            [Parameter(x) for x in [Q, G, h, A, b]]

    def forward(self, inputs):
        return QPFunction()(inputs, self.Q, self.p, self.G, self.h, self.A, self.b)
