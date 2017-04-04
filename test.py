#!/usr/bin/env python3
#
# Run these tests with: nosetests -v -d <file>.py
#   This will run all functions even if one throws an assertion.
#
# For debugging: ./<file>.py
#   Easier to print statements.
#   This will exit qfter the first assertion.

import os

import torch
from torch.autograd import Function, Variable

import numpy as np
import numpy.random as npr
import numpy.testing as npt
np.set_printoptions(precision=6)

import numdifftools as nd
import cvxpy as cp

# from solver import BlockSolver as Solver

from nose.tools import with_setup, assert_almost_equal

import sys
sys.path.append('..')
import qpth
import qpth.solvers.cvxpy as qp_cvxpy

import qpth.solvers.pdipm.single as pdipm_s
import qpth.solvers.pdipm.batch as pdipm_b

from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

ATOL=1e-2
RTOL=1e-4

cuda = True
verbose = True

def get_grads(nBatch=1, nz=10, neq=1, nineq=3, Qscale=1.,
              Gscale=1., hscale=1., Ascale=1., bscale=1.):
    assert(nBatch==1)
    npr.seed(1)
    L = np.random.randn(nz,nz)
    Q = Qscale*L.dot(L.T)
    G = Gscale*npr.randn(nineq,nz)
    # h = hscale*npr.randn(nineq)
    z0 = npr.randn(nz)
    s0 = npr.rand(nineq)
    h = G.dot(z0)+s0
    A = Ascale*npr.randn(neq,nz)
    # b = bscale*npr.randn(neq)
    b = A.dot(z0)

    p = npr.randn(nBatch,nz)
    # print(np.linalg.norm(p))
    truez = npr.randn(nBatch,nz)

    Q, p, G, h, A, b, truez = [x.astype(np.float64) for x in [Q, p, G, h, A, b, truez]]
    zhat, nu, lam = qp_cvxpy.forward_single_np(p[0], Q, G, h, A, b)

    grads = get_grads_torch(Q, p, G, h, A, b, truez)
    return [p[0], Q, G, h, A, b, truez], grads

def get_grads_torch(Q, p, G, h, A, b, truez):
    Q, p, G, h, A, b, truez = [torch.DoubleTensor(x) for x in [Q, p, G, h, A, b, truez]]
    if cuda:
        Q, p, G, h, A, b, truez = [x.cuda() for x in [Q, p, G, h, A, b, truez]]

    Q, p, G, h, A, b = [Variable(x) for x in [Q, p, G, h, A, b]]
    for x in [Q, p, G, h]: x.requires_grad = True

    nBatch = 1
    if b.nelement() > 0:
        A.requires_grad = True
        b.requires_grad = True

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
        zhat, nu, lam = qp_cvxpy.forward_single_np(p, Q, G, h, A, b)
        return 0.5*np.sum(np.square(zhat - truez))

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
        G = G.reshape(nineq,nz)
        zhat, nu, lam = qp_cvxpy.forward_single_np(p, Q, G, h, A, b)
        return 0.5*np.sum(np.square(zhat - truez))

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
        zhat, nu, lam = qp_cvxpy.forward_single_np(p, Q, G, h, A, b)
        return 0.5*np.sum(np.square(zhat - truez))

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
        A = A.reshape(neq,nz)
        zhat, nu, lam = qp_cvxpy.forward_single_np(p, Q, G, h, A, b)
        return 0.5*np.sum(np.square(zhat - truez))

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
        zhat, nu, lam = qp_cvxpy.forward_single_np(p, Q, G, h, A, b)
        return 0.5*np.sum(np.square(zhat - truez))

    df = nd.Gradient(f)
    db_fd = df(b)
    if verbose:
        print('db_fd: ', db_fd)
        print('db: ', db)
    npt.assert_allclose(db_fd, db, rtol=RTOL, atol=ATOL)


if __name__=='__main__':
    test_dl_dp()
    test_dl_dG()
    test_dl_dh()
    test_dl_dA()
    test_dl_db()
