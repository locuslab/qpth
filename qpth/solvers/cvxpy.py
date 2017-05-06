import cvxpy as cp
import numpy as np
import torch


def forward_single_np(Q, p, G, h, A, b):
    nz, neq, nineq = p.shape[0], A.shape[0] if A is not None else 0, G.shape[0]

    z_ = cp.Variable(nz)

    obj = cp.Minimize(0.5 * cp.quad_form(z_, Q) + p.T * z_)
    eqCon = A * z_ == b if neq > 0 else None
    if nineq > 0:
        slacks = cp.Variable(nineq)
        ineqCon = G * z_ + slacks == h
        slacksCon = slacks >= 0
    else:
        ineqCon = slacks = slacksCon = None
    cons = [x for x in [eqCon, ineqCon, slacksCon] if x is not None]
    prob = cp.Problem(obj, cons)
    prob.solve() #solver=cp.SCS, max_iters=5000, verbose=False)
    # prob.solve(solver=cp.SCS, max_iters=10000, verbose=True)
    assert('optimal' in prob.status)
    zhat = np.array(z_.value).ravel()
    nu = np.array(eqCon.dual_value).ravel() if eqCon is not None else None
    if ineqCon is not None:
        lam = np.array(ineqCon.dual_value).ravel()
        slacks = np.array(slacks.value).ravel()
    else:
        lam = slacks = None

    return prob.value, zhat, nu, lam, slacks


def forward_cvxpy(input, L, G, A, z0, s0):
    assert False
    start = time.time()
    in_np, L_np, G_np, A_np, z0_np, s0_np = [
        toNp(v) for v in [input, L, G, A, z0, s0]]

    neq = A_np.shape[0]
    nineq = G_np.shape[0]
    assert(neq > 0 or nineq > 0)

    assert(input.dim() == 2)
    nBatch, nz = input.size()
    self.neq, self.nineq, self.nz = neq, nineq, nz
    zhats, nus, lams = [], [], []
    for i in range(nBatch):
        zhat_i, nu_i, lam_i = self.forward_single_np(in_np[i], L_np, G_np,
                                                     A_np, z0_np, s0_np)
        zhats.append(zhat_i)
        nus.append(nu_i)
        lams.append(lam_i)
    zhats = torch.Tensor(np.array(zhats)).type_as(L)

    self.nus = torch.Tensor(np.array(nus)) if neq > 0 else None
    self.lams = torch.Tensor(np.array(lams)) if nineq > 0 else None

    self.save_for_backward(input, zhats,
                           L, G, A, z0, s0)
    print('  + Forward pass took {:0.4f} seconds.'.format(time.time() - start))
    return zhats
