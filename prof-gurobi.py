#!/usr/bin/env python3

import argparse
import sys

import numpy as np
import numpy.random as npr

# import qpth.solvers.pdipm.single as pdipm_s
import qpth.solvers.pdipm.batch as pdipm_b

import itertools
import time

import torch

# import gurobipy as gpy

from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
                                     color_scheme='Linux', call_pdb=1)

import setproctitle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nTrials', type=int, default=10)
    args = parser.parse_args()
    setproctitle.setproctitle('bamos.optnet.prof-gurobi')

    npr.seed(0)

    prof(args)


def prof(args):
    print('|  \# Vars | \# Batch | Gurobi | single | batched |')
    print('|----------+----------+--------+--------+---------|')
    # for nz, nBatch in itertools.product([100,500], [1, 64, 128]):
    for nz, nBatch in itertools.product([100], [1, 64, 128]):
        times = []
        for i in range(args.nTrials):
            times.append(prof_instance(nz, nBatch))
        times = np.array(times)
        print(("| {:5d} " * 2 + "| ${:.5e} \pm {:.5e}$ s " * 3 + '|').format(
            *([nz, nBatch] + [item for sublist in zip(times.mean(axis=0), times.std(axis=0))
                              for item in sublist])))


def prof_instance(nz, nBatch, cuda=True):
    nineq, neq = 100, 0
    assert(neq == 0)
    L = npr.rand(nBatch, nz, nz)
    Q = np.matmul(L, L.transpose((0, 2, 1))) + 1e-3 * np.eye(nz, nz)
    G = npr.randn(nBatch, nineq, nz)
    z0 = npr.randn(nBatch, nz)
    s0 = npr.rand(nBatch, nineq)
    p = npr.randn(nBatch, nz)
    h = np.matmul(G, np.expand_dims(z0, axis=(2))).squeeze(2) + s0
    A = npr.randn(nBatch, neq, nz)
    b = np.matmul(A, np.expand_dims(z0, axis=(2))).squeeze(2)

    # zhat_g = []
    # gurobi_time = 0.0
    # for i in range(nBatch):
    #     m = gpy.Model()
    #     zhat = m.addVars(nz, lb=-gpy.GRB.INFINITY, ub=gpy.GRB.INFINITY)

    #     obj = 0.0
    #     for j in range(nz):
    #         for k in range(nz):
    #             obj += 0.5 * Q[i, j, k] * zhat[j] * zhat[k]
    #         obj += p[i, j] * zhat[j]
    #     m.setObjective(obj)
    #     for j in range(nineq):
    #         con = 0
    #         for k in range(nz):
    #             con += G[i, j, k] * zhat[k]
    #         m.addConstr(con <= h[i, j])
    #     m.setParam('OutputFlag', False)
    #     start = time.time()
    #     m.optimize()
    #     gurobi_time += time.time() - start
    #     t = np.zeros(nz)
    #     for j in range(nz):
    #         t[j] = zhat[j].x
    # zhat_g.append(t)
    gurobi_time = -1

    p, L, Q, G, z0, s0, h = [torch.Tensor(x) for x in [p, L, Q, G, z0, s0, h]]
    if cuda:
        p, L, Q, G, z0, s0, h = [x.cuda() for x in [p, L, Q, G, z0, s0, h]]
    if neq > 0:
        A = torch.Tensor(A)
        b = torch.Tensor(b)
    else:
        A, b = [torch.Tensor()] * 2
    if cuda:
        A = A.cuda()
        b = b.cuda()

    # af = adact.AdactFunction()

    # single_results = []
    start = time.time()
    # for i in range(nBatch):
    # A_i = A[i] if neq > 0 else A
    # b_i = b[i] if neq > 0 else b
    # U_Q, U_S, R = pdipm_s.pre_factor_kkt(Q[i], G[i], A_i)
    # single_results.append(pdipm_s.forward(p[i], Q[i], G[i], A_i, b_i, h[i],
    #                                       U_Q, U_S, R))
    single_time = time.time() - start

    start = time.time()
    Q_LU, S_LU, R = pdipm_b.pre_factor_kkt(Q, G, A)
    zhat_b, nu_b, lam_b, s_b = pdipm_b.forward(Q, p, G, h, A, b, Q_LU, S_LU, R)
    batched_time = time.time() - start

    # Usually between 1e-4 and 1e-5:
    # print('Diff between gurobi and pdipm: ',
    #       np.linalg.norm(zhat_g[0]-zhat_b[0].cpu().numpy()))
    # import IPython, sys; IPython.embed(); sys.exit(-1)

    # import IPython, sys; IPython.embed(); sys.exit(-1)
    # zhat_diff = (single_results[0][0] - zhat_b[0]).norm()
    # lam_diff = (single_results[0][2] - lam_b[0]).norm()
    # eps = 0.1 # Pretty relaxed.
    # if zhat_diff > eps or lam_diff > eps:
    #     print('===========')
    #     print("Warning: Single and batched solutions might not match.")
    #     print("  + zhat_diff: {}".format(zhat_diff))
    #     print("  + lam_diff: {}".format(lam_diff))
    #     print("  + (nz, neq, nineq, nBatch) = ({}, {}, {}, {})".format(
    #         nz, neq, nineq, nBatch))
    #     print('===========')

    return gurobi_time, single_time, batched_time


if __name__ == '__main__':
    main()
