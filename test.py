#!/usr/bin/env python3
#
# Run these tests with: nosetests -v -d <file>.py
#   This will run all functions even if one throws an assertion.
#
# For debugging: ./<file>.py
#   Easier to print statements.
#   This will exit qfter the first assertion.

import torch
from torch.autograd import Variable

import numpy as np
import numpy.random as npr
import numpy.testing as npt
from numpy.testing import dec
np.set_printoptions(precision=6)

import numdifftools as nd

import sys
sys.path.append('..')
import qpth
from qpth.util import bdiag, expandParam, extract_nBatch
import qpth.solvers.cvxpy as qp_cvxpy

# import qpth.solvers.pdipm.single as pdipm_s
import qpth.solvers.pdipm.batch as pdipm_b
import qpth.solvers.pdipm.spbatch as pdipm_spb

# from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(mode='Verbose',
#      color_scheme='Linux', call_pdb=1)

ATOL = 1e-2
RTOL = 1e-4

cuda = False
verbose = True


def get_grads(nBatch=1, nz=10, neq=1, nineq=3, Qscale=1.,
              Gscale=1., hscale=1., Ascale=1., bscale=1.):
    assert(nBatch == 1)
    npr.seed(1)
    L = np.random.randn(nz, nz)
    Q = Qscale * L.dot(L.T)
    G = Gscale * npr.randn(nineq, nz)
    # h = hscale*npr.randn(nineq)
    z0 = npr.randn(nz)
    s0 = npr.rand(nineq)
    h = G.dot(z0) + s0
    A = Ascale * npr.randn(neq, nz)
    # b = bscale*npr.randn(neq)
    b = A.dot(z0)

    p = npr.randn(nBatch, nz)
    # print(np.linalg.norm(p))
    truez = npr.randn(nBatch, nz)

    Q, p, G, h, A, b, truez = [x.astype(np.float64) for x in
                               [Q, p, G, h, A, b, truez]]
    _, zhat, nu, lam, slacks = qp_cvxpy.forward_single_np(Q, p[0], G, h, A, b)

    grads = get_grads_torch(Q, p, G, h, A, b, truez)
    return [p[0], Q, G, h, A, b, truez], grads


def get_grads_torch(Q, p, G, h, A, b, truez):
    Q, p, G, h, A, b, truez = [
        torch.DoubleTensor(x) if len(x) > 0 else torch.DoubleTensor()
        for x in [Q, p, G, h, A, b, truez]]
    if cuda:
        Q, p, G, h, A, b, truez = [x.cuda() for x in [Q, p, G, h, A, b, truez]]

    Q, p, G, h, A, b = [Variable(x) for x in [Q, p, G, h, A, b]]
    for x in [Q, p, G, h]:
        x.requires_grad = True

    # nBatch = 1
    if b.nelement() > 0:
        A.requires_grad = True
        b.requires_grad = True

    # zhats = qpth.qp.QPFunction(solver=qpth.qp.QPSolvers.CVXPY)(Q, p, G, h, A, b)
    zhats = qpth.qp.QPFunction()(Q, p, G, h, A, b)

    dl_dzhat = zhats.data - truez
    zhats.backward(dl_dzhat)

    grads = [x.grad.data.squeeze(0).cpu().numpy() for x in [Q, p, G, h]]
    if A.nelement() > 0:
        grads += [x.grad.data.squeeze(0).cpu().numpy() for x in [A, b]]
    else:
        grads += [None, None]
    return grads


def test_dl_dp():
    nz, neq, nineq = 10, 2, 3
    [p, Q, G, h, A, b, truez], [dQ, dp, dG, dh, dA, db] = get_grads(
        nz=nz, neq=neq, nineq=nineq, Qscale=100., Gscale=100., Ascale=100.)

    def f(p):
        _, zhat, nu, lam, slacks = qp_cvxpy.forward_single_np(Q, p, G, h, A, b)
        return 0.5 * np.sum(np.square(zhat - truez))

    df = nd.Gradient(f)
    dp_fd = df(p)
    if verbose:
        print('dp_fd: ', dp_fd)
        print('dp: ', dp)
    npt.assert_allclose(dp_fd, dp, rtol=RTOL, atol=ATOL)


def test_dl_dG():
    nz, neq, nineq = 10, 0, 3
    [p, Q, G, h, A, b, truez], [dQ, dp, dG, dh, dA, db] = get_grads(
        nz=nz, neq=neq, nineq=nineq)

    def f(G):
        G = G.reshape(nineq, nz)
        _, zhat, nu, lam, slacks = qp_cvxpy.forward_single_np(Q, p, G, h, A, b)
        return 0.5 * np.sum(np.square(zhat - truez))

    df = nd.Gradient(f)
    dG_fd = df(G.ravel()).reshape(nineq, nz)
    if verbose:
        # print('dG_fd[1,:]: ', dG_fd[1,:])
        # print('dG[1,:]: ', dG[1,:])
        print('dG_fd: ', dG_fd)
        print('dG: ', dG)
    npt.assert_allclose(dG_fd, dG, rtol=RTOL, atol=ATOL)


def test_dl_dh():
    nz, neq, nineq = 10, 0, 3
    [p, Q, G, h, A, b, truez], [dQ, dp, dG, dh, dA, db] = get_grads(
        nz=nz, neq=neq, nineq=nineq, Qscale=1., Gscale=1.)

    def f(h):
        _, zhat, nu, lam, slacks = qp_cvxpy.forward_single_np(Q, p, G, h, A, b)
        return 0.5 * np.sum(np.square(zhat - truez))

    df = nd.Gradient(f)
    dh_fd = df(h)
    if verbose:
        print('dh_fd: ', dh_fd)
        print('dh: ', dh)
    npt.assert_allclose(dh_fd, dh, rtol=RTOL, atol=ATOL)


def test_dl_dA():
    nz, neq, nineq = 10, 3, 1
    [p, Q, G, h, A, b, truez], [dQ, dp, dG, dh, dA, db] = get_grads(
        nz=nz, neq=neq, nineq=nineq, Qscale=100., Gscale=100., Ascale=100.)

    def f(A):
        A = A.reshape(neq, nz)
        _, zhat, nu, lam, slacks = qp_cvxpy.forward_single_np(Q, p, G, h, A, b)
        return 0.5 * np.sum(np.square(zhat - truez))

    df = nd.Gradient(f)
    dA_fd = df(A.ravel()).reshape(neq, nz)
    if verbose:
        # print('dA_fd[0,:]: ', dA_fd[0,:])
        # print('dA[0,:]: ', dA[0,:])
        print('dA_fd: ', dA_fd)
        print('dA: ', dA)
    npt.assert_allclose(dA_fd, dA, rtol=RTOL, atol=ATOL)


def test_dl_db():
    nz, neq, nineq = 10, 3, 1
    [p, Q, G, h, A, b, truez], [dQ, dp, dG, dh, dA, db] = get_grads(
        nz=nz, neq=neq, nineq=nineq, Qscale=100., Gscale=100., Ascale=100.)

    def f(b):
        _, zhat, nu, lam, slacks = qp_cvxpy.forward_single_np(Q, p, G, h, A, b)
        return 0.5 * np.sum(np.square(zhat - truez))

    df = nd.Gradient(f)
    db_fd = df(b)
    if verbose:
        print('db_fd: ', db_fd)
        print('db: ', db)
    npt.assert_allclose(db_fd, db, rtol=RTOL, atol=ATOL)


def get_kkt_problem():
    def cast(m):
        # return m.cuda().double()
        return m.double()

    nBatch, nx, nineq, neq = 2, 5, 4, 3
    Q = cast(torch.randn(nx, nx))
    Q = Q.mm(Q.t())
    p = cast(torch.randn(nx))
    G = cast(torch.randn(nBatch, nineq, nx))
    h = cast(torch.zeros(nBatch, nineq))
    A = cast(torch.randn(neq, nx))
    b = cast(torch.randn(neq))

    nBatch = extract_nBatch(Q, p, G, h, A, b)
    Q, _ = expandParam(Q, nBatch, 3)
    p, _ = expandParam(p, nBatch, 2)
    G, _ = expandParam(G, nBatch, 3)
    h, _ = expandParam(h, nBatch, 2)
    A, _ = expandParam(A, nBatch, 3)
    b, _ = expandParam(b, nBatch, 2)

    d = torch.rand(nBatch, nineq).type_as(Q)
    D = bdiag(d)
    rx = torch.rand(nBatch, nx).type_as(Q)
    rs = torch.rand(nBatch, nineq).type_as(Q)
    rz = torch.rand(nBatch, nineq).type_as(Q)
    ry = torch.rand(nBatch, neq).type_as(Q)

    return Q, p, G, h, A, b, d, D, rx, rs, rz, ry


def test_lu_kkt_solver():
    Q, p, G, h, A, b, d, D, rx, rs, rz, ry = get_kkt_problem()

    dx, ds, dz, dy = pdipm_b.factor_solve_kkt(Q, D, G, A, rx, rs, rz, ry)

    Q_LU, S_LU, R = pdipm_b.pre_factor_kkt(Q, G, A)
    pdipm_b.factor_kkt(S_LU, R, d)
    dx_, ds_, dz_, dy_ = pdipm_b.solve_kkt(Q_LU, d, G, A, S_LU, rx, rs, rz, ry)

    npt.assert_allclose(dx.numpy(), dx_.numpy(), rtol=RTOL, atol=ATOL)
    npt.assert_allclose(ds.numpy(), ds_.numpy(), rtol=RTOL, atol=ATOL)
    npt.assert_allclose(dz.numpy(), dz_.numpy(), rtol=RTOL, atol=ATOL)
    npt.assert_allclose(dy.numpy(), dy_.numpy(), rtol=RTOL, atol=ATOL)


def test_ir_kkt_solver():
    Q, p, G, h, A, b, d, D, rx, rs, rz, ry = get_kkt_problem()

    dx, ds, dz, dy = pdipm_b.factor_solve_kkt(Q, D, G, A, rx, rs, rz, ry)
    dx_, ds_, dz_, dy_ = pdipm_b.solve_kkt_ir(
        Q, D, G, A, rx, rs, rz, ry, niter=1)

    npt.assert_allclose(dx.numpy(), dx_.numpy(), rtol=RTOL, atol=ATOL)
    npt.assert_allclose(ds.numpy(), ds_.numpy(), rtol=RTOL, atol=ATOL)
    npt.assert_allclose(dz.numpy(), dz_.numpy(), rtol=RTOL, atol=ATOL)
    npt.assert_allclose(dy.numpy(), dy_.numpy(), rtol=RTOL, atol=ATOL)


@npt.dec.skipif(
    not torch.cuda.is_available() or not hasattr(torch, 'spbqrfactsolve'))
def test_sparse_forward():
    torch.manual_seed(0)

    nBatch, nx, nineq, neq = 2, 5, 4, 3

    def cast(m):
        return m.cuda().double()

    spTensor = torch.cuda.sparse.DoubleTensor
    iTensor = torch.cuda.LongTensor

    Qi = iTensor([range(nx), range(nx)])
    Qv = cast(torch.ones(nBatch, nx))
    Qsz = torch.Size([nx, nx])
    Q0 = spTensor(Qi, Qv[0], Qsz)

    Gi = iTensor([range(nineq), range(nineq)])
    Gv = cast(torch.randn(nBatch, nineq))
    Gsz = torch.Size([nineq, nx])
    G0 = spTensor(Gi, Gv[0], Gsz)
    h = cast(torch.randn(nBatch, nineq))

    Ai = iTensor([range(neq), range(neq)])
    Av = Gv[:, :neq].clone()
    Asz = torch.Size([neq, nx])
    A0 = spTensor(Ai, Av[0], Asz)
    b = h[:, :neq].clone()

    p = cast(torch.randn(nBatch, nx))

    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(mode='Verbose',
                                         color_scheme='Linux', call_pdb=1)
    xhats0_cp = qpth.qp.QPFunction(solver=qpth.qp.QPSolvers.CVXPY)(
        *[Variable(y) for y in
          [Q0.to_dense(), p[0], G0.to_dense(), h[0], A0.to_dense(), b[0]]]).data.squeeze()

    xhats, nus, lams, slacks = pdipm_spb.forward(Qi, Qv, Qsz, p, Gi, Gv, Gsz, h,
                                                 Ai, Av, Asz, b, verbose=-1,
                                                 notImprovedLim=3, maxIter=20)
    npt.assert_allclose(xhats0_cp.cpu().numpy(),
                        xhats[0].cpu().numpy(), rtol=RTOL, atol=ATOL)

    Qv, p, Gv, h, Av, b = [Variable(x) for x in [Qv, p, Gv, h, Av, b]]
    xhats_qpf = qpth.qp.SpQPFunction(Qi, Qsz, Gi, Gsz, Ai, Asz)(
        Qv, p, Gv, h, Av, b
    ).data
    npt.assert_allclose(xhats.cpu().numpy(),
                        xhats_qpf.cpu().numpy(), rtol=RTOL, atol=ATOL)


@npt.dec.skipif(
    not torch.cuda.is_available() or not hasattr(torch, 'spbqrfactsolve'))
def test_sparse_backward():
    torch.manual_seed(0)

    nBatch, nx, nineq, neq = 1, 5, 4, 3

    def cast(m):
        return m.cuda().double()

    spTensor = torch.cuda.sparse.DoubleTensor
    iTensor = torch.cuda.LongTensor

    Qi = iTensor([range(nx), range(nx)])
    Qv = cast(torch.ones(nBatch, nx))
    Qsz = torch.Size([nx, nx])
    Q0 = spTensor(Qi, Qv[0], Qsz)

    Gi = iTensor([range(nineq), range(nineq)])
    Gv = cast(torch.randn(nBatch, nineq))
    Gsz = torch.Size([nineq, nx])
    G0 = spTensor(Gi, Gv[0], Gsz)
    h = cast(torch.randn(nBatch, nineq))

    Ai = iTensor([range(neq), range(neq)])
    Av = Gv[:, :neq].clone()
    Asz = torch.Size([neq, nx])
    A0 = spTensor(Ai, Av[0], Asz)
    b = h[:, :neq].clone()

    p = cast(torch.randn(nBatch, nx))
    truex = Variable(cast(torch.randn(nBatch, nx)))

    Qv, p, Gv, h, Av, b = [Variable(x) for x in [Qv, p, Gv, h, Av, b]]
    for x in [Qv, p, Gv, h, Av, b]:
        x.requires_grad = True
    xhats = qpth.qp.SpQPFunction(Qi, Qsz, Gi, Gsz, Ai, Asz)(
        Qv, p, Gv, h, Av, b
    )
    loss = torch.norm(xhats - truex)
    loss.backward()

    # dQv, dGv, dAv = Qv.grad, Gv.grad, Av.grad
    dQv = Qv.grad

    Q0 = Q0.to_dense()
    p0 = p[0].data
    G0 = G0.to_dense()
    h0 = h[0].data
    A0 = A0.to_dense()
    b0 = b[0].data
    Q0, p0, G0, h0, A0, b0 = [Variable(y) for y in [Q0, p0, G0, h0, A0, b0]]
    for x in [Q0, p0, G0, h0, A0, b0]:
        x.requires_grad = True
    xhats_dense = qpth.qp.QPFunction()(Q0, p0, G0, h0, A0, b0)
    loss_dense = torch.norm(xhats_dense - truex)
    loss_dense.backward()

    # dQ, dG, dA = Q0.grad, G0.grad, A0.grad
    dQ = Q0.grad

    npt.assert_allclose(dQv.squeeze().data.cpu().numpy(), dQ.data.diag().cpu().numpy(),
                        rtol=RTOL, atol=ATOL)
    # TODO: dG/dGv don't match
    # TODO: dA/dAv don't match


if __name__ == '__main__':
    test_dl_dp()
    test_dl_dG()
    test_dl_dh()
    test_dl_dA()
    test_dl_db()
    test_lu_kkt_solver()
    test_ir_kkt_solver()
    # test_sparse_forward()
    # test_sparse_backward()
