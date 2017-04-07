import torch
from torch.autograd import Function

from .util import bger, expandParam, extract_nBatch
from .solvers.pdipm import batch as pdipm_b
# from .solvers.pdipm import single as pdipm_s


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
        return zhats

    def backward(self, dl_dzhat):
        zhats, Q, p, G, h, A, b = self.saved_tensors
        nBatch = extract_nBatch(Q, p, G, h, A, b)
        Q, Q_e = expandParam(Q, nBatch, 3)
        p, p_e = expandParam(p, nBatch, 2)
        G, G_e = expandParam(G, nBatch, 3)
        h, h_e = expandParam(h, nBatch, 2)
        A, A_e = expandParam(A, nBatch, 3)
        b, b_e = expandParam(b, nBatch, 2)

        # neq, nineq, nz = self.neq, self.nineq, self.nz
        neq, nineq = self.neq, self.nineq

        d = self.lams / self.slacks
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
        dQs = 0.5 * (bger(dx, zhats) + bger(zhats, dx))
        if Q_e:
            dQs = dQs.mean(0).squeeze(0)

        grads = (dQs, dps, dGs, dhs, dAs, dbs)

        return grads
