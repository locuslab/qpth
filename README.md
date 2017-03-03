# qpth • [ ![Release] [release-image] ] [releases] [ ![License] [license-image] ] [license]

*A fast and differentiable QP solver for PyTorch.
Crafted by [Brandon Amos](http://bamos.github.io) and
[J. Zico Kolter](http://zicokolter.com).*

[release-image]: http://img.shields.io/badge/release-0.0.2-blue.svg?style=flat
[releases]: https://pypi.python.org/pypi/qpth

[license-image]: http://img.shields.io/badge/license-Apache--2-blue.svg?style=flat
[license]: LICENSE

# Optimization primitives are important for modern (deep) machine learning.

[Mathematical optimization](https://en.wikipedia.org/wiki/Mathematical_optimization)
is a well-studied language of expressing solutions to many real-life problems
that come up in machine learning and many other fields such as mechanics,
economics, EE, operations research, control engineering, geophysics,
and molecular modeling.
As we build our machine learning systems to interact with real
data from these fields, we often **cannot** (but sometimes can)
simply "learn away" the optimization sub-problems by adding more
layers in our network. Well-defined optimization problems may be added
if you have a thorough understanding of your feature space, but
oftentimes we **don't** have this understanding and resort to
automatic feature learning for our tasks.

Until this repository, **no** modern deep learning library has provided
a way of adding a learnable optimization layer (other than simply unrolling
an optimization procedure, which is inefficient and inexact) into
our model formulation that we can quickly try to see if it's a nice way
of expressing our data.

See our paper
[OptNet: Differentiable Optimization as a Layer in Neural Networks](https://arxiv.org/abs/1703.00443)
and code at
[locuslab/optnet](https://github.com/locuslab/optnet)
if you are interested in learning more about our initial exploration
in this space of automatically learning quadratic program layers
for signal denoising and sudoku.

# What is a quadratic program (QP) layer?

[Wikipedia gives a great introduction to quadratic programming](https://en.wikipedia.org/wiki/Quadratic_programming).

We define a quadratic program layer as

<p align="center"><img src="https://rawgit.com/locuslab/qpth/master/svgs/bf7a49b8e29e8140dec6067072be2426.svg?invert_in_darkmode" align=middle width=269.8014pt height=82.140795pt/></p>
where <img src="https://rawgit.com/locuslab/qpth/master/svgs/386deda277c0c42d998cbe67116467ef.svg?invert_in_darkmode" align=middle width=70.173015pt height=21.88065pt/> is the current layer,
<img src="https://rawgit.com/locuslab/qpth/master/svgs/c0c08c72c0e0ce3f65f5dbc0f627dc71.svg?invert_in_darkmode" align=middle width=53.529135pt height=21.88065pt/> is the previous layer,
<img src="https://rawgit.com/locuslab/qpth/master/svgs/53dd8a31c0c73ead2d01494377ef1a7a.svg?invert_in_darkmode" align=middle width=48.77928pt height=21.88065pt/> is the optimization variable,
and
<img src="https://rawgit.com/locuslab/qpth/master/svgs/ad3854b968da287b3c9ce4b6cf4de374.svg?invert_in_darkmode" align=middle width=97.71003pt height=25.40967pt/>,
<img src="https://rawgit.com/locuslab/qpth/master/svgs/b227a7b9212d56d91ff063567643ac18.svg?invert_in_darkmode" align=middle width=74.585115pt height=23.88969pt/>,
<img src="https://rawgit.com/locuslab/qpth/master/svgs/1d25dc7bc81ab504c52424d70a1ad0c4.svg?invert_in_darkmode" align=middle width=100.582185pt height=25.40967pt/>,
<img src="https://rawgit.com/locuslab/qpth/master/svgs/e3fb7fd8febcf9d51fca443f6193eef3.svg?invert_in_darkmode" align=middle width=76.90815pt height=23.88969pt/>,
<img src="https://rawgit.com/locuslab/qpth/master/svgs/1cc444e2ce2708b3d72253707732eb23.svg?invert_in_darkmode" align=middle width=96.28971pt height=25.40967pt/>, and
<img src="https://rawgit.com/locuslab/qpth/master/svgs/c9e3c70b2ea30a8310acf540b2eaa43e.svg?invert_in_darkmode" align=middle width=74.43612pt height=23.88969pt/> are parameters of
the optimization problem.
As the notation suggests, these parameters can depend in any differentiable way
on the previous layer <img src="https://rawgit.com/locuslab/qpth/master/svgs/6af8e9329c416994c3690752bde99a7d.svg?invert_in_darkmode" align=middle width=12.61788pt height=13.38744pt/>, and which can be optimized just like
any other weights in a neural network.
For simplicity, we often drop the explicit dependence on
<img src="https://rawgit.com/locuslab/qpth/master/svgs/6af8e9329c416994c3690752bde99a7d.svg?invert_in_darkmode" align=middle width=12.61788pt height=13.38744pt/> from the parameters.

# What does this library provide?

This library provides a fast, batched, and differentiable
QP layer as a PyTorch Function.

# How fast is this compared to Gurobi?

![](./images/timing.png)

*Performance of the Gurobi (red), qpth single (ours, blue),
qpth batched (ours, green) solvers.*

We run our solver on an unloaded Titan X GPU and Gurobi on an
unloaded quad-core Intel Core i7-5960X CPU @ 3.00GHz.
We set up the same random QP across all three frameworks and vary the number of
variable, constraints, and batch size.

*Experimental details:* we sample
entries of a matrix <img src="https://rawgit.com/locuslab/qpth/master/svgs/6bac6ec50c01592407695ef84f457232.svg?invert_in_darkmode" align=middle width=13.33827pt height=21.69783pt/> from a random uniform distribution and set <img src="https://rawgit.com/locuslab/qpth/master/svgs/8c3321a339ffd90c61547bcd938c775f.svg?invert_in_darkmode" align=middle width=134.31693pt height=26.88906pt/>, sample <img src="https://rawgit.com/locuslab/qpth/master/svgs/5201385589993766eea584cd3aa6fa13.svg?invert_in_darkmode" align=middle width=13.247025pt height=21.69783pt/> with random normal entries, and set <img src="https://rawgit.com/locuslab/qpth/master/svgs/2ad9d098b937e46f9f58968551adac57.svg?invert_in_darkmode" align=middle width=9.79341pt height=22.06347pt/> by
selecting generating some <img src="https://rawgit.com/locuslab/qpth/master/svgs/d1a81d9dc6dd30e43ba27c5490a34a32.svg?invert_in_darkmode" align=middle width=14.519505pt height=13.38744pt/> random normal and <img src="https://rawgit.com/locuslab/qpth/master/svgs/ac3148a5746b81298cb0c456b661f197.svg?invert_in_darkmode" align=middle width=14.58039pt height=13.38744pt/> random uniform and
setting <img src="https://rawgit.com/locuslab/qpth/master/svgs/deb38135ff51e48d01add7f5728994d2.svg?invert_in_darkmode" align=middle width=94.003965pt height=22.06347pt/> (we didn't include equality constraints just for
simplicity, and since the number of inequality constraints in the primary
driver of complexity for the iterations in a primal-dual interior point
method. The choice of <img src="https://rawgit.com/locuslab/qpth/master/svgs/2ad9d098b937e46f9f58968551adac57.svg?invert_in_darkmode" align=middle width=9.79341pt height=22.06347pt/> guarantees the problem is feasible.

The figure above shows the means and standard deviations
of running each trial 10 times, showing that our solver
outperforms Gurobi, itself a highly tuned solver, in all batched instances.
For the minibatch size of 128, we solve all problems in an average of 0.18
seconds, whereas Gurobi tasks an average of 4.7 seconds.  In the context of
training a deep architecture this type of speed difference for a single
minibatch can make the difference between a practical and a completely unusable
solution.

# Setup and Dependencies

+ Python/numpy
+ [PyTorch](https://pytorch.org)
  + The code currently requires a source install from the master branch from
    [our fork](https://github.com/locuslab/pytorch) for new batch triangular
    factorization functions we have added.
    We are currently working with the PyTorch team to get these new features
    merged into Torch proper.
+ [bamos/block](https://github.com/bamos/block):
  *Our intelligent block matrix library for numpy, PyTorch, and beyond.*

## Install via pip

```
pip install qpth
```

# Usage

You can see many full working examples in our
[locuslab/optnet](https://github.com/locuslab/optnet)
repo.

Here's an example that adds a small QP layer with
only inequality constraints at the end of a
fully-connected network.
This layer has <img src="https://rawgit.com/locuslab/qpth/master/svgs/1050e0281ea114a4ffef040ed805b874.svg?invert_in_darkmode" align=middle width=103.244955pt height=26.88906pt/> where <img src="https://rawgit.com/locuslab/qpth/master/svgs/ddcb483302ed36a59286424aa5e0be17.svg?invert_in_darkmode" align=middle width=11.509575pt height=21.69783pt/>
is a lower-triangular matrix and <img src="https://rawgit.com/locuslab/qpth/master/svgs/f490d9b8b6304264612ac7d32bebfc61.svg?invert_in_darkmode" align=middle width=94.003965pt height=22.06347pt/>
for some learnable <img src="https://rawgit.com/locuslab/qpth/master/svgs/d1a81d9dc6dd30e43ba27c5490a34a32.svg?invert_in_darkmode" align=middle width=14.519505pt height=13.38744pt/> and <img src="https://rawgit.com/locuslab/qpth/master/svgs/ac3148a5746b81298cb0c456b661f197.svg?invert_in_darkmode" align=middle width=14.58039pt height=13.38744pt/> to ensure the
problem is always feasible.

```Python
from qpth.qp import QPFunction

...

class OptNet(nn.Module):
    def __init__(self, nFeatures, nHidden, nCls, bn, nineq=200, neq=0, eps=1e-4):
        super().__init__()

        self.nFeatures = nFeatures
        self.nHidden = nHidden
        self.bn = bn
        self.nCls = nCls
        self.nineq = nineq
        self.neq = neq
        self.eps = eps

        if bn:
            self.bn1 = nn.BatchNorm1d(nHidden)
            self.bn2 = nn.BatchNorm1d(nCls)

        self.fc1 = nn.Linear(nFeatures, nHidden)
        self.fc2 = nn.Linear(nHidden, nCls)

        self.M = Variable(torch.tril(torch.ones(nCls, nCls)).cuda())
        self.L = Parameter(torch.tril(torch.rand(nCls, nCls).cuda()))
        self.G = Parameter(torch.Tensor(nineq,nCls).uniform_(-1,1).cuda())
        self.z0 = Parameter(torch.zeros(nCls).cuda())
        self.s0 = Parameter(torch.ones(nineq).cuda())

    def forward(self, x):
        nBatch = x.size(0)

        # FC-ReLU-(BN)-FC-ReLU-(BN)-QP-Softmax
        x = x.view(nBatch, -1)
        x = F.relu(self.fc1(x))
        if self.bn:
            x = self.bn1(x)
        x = F.relu(self.fc2(x))
        if self.bn:
            x = self.bn2(x)

        L = self.M*self.L
        Q = L.mm(L.t()) + self.eps*Variable(torch.eye(self.nCls)).cuda()
        h = self.G.mv(self.z0)+self.s0
        e = Variable(torch.Tensor())
        x = QPFunction(verbose=False)(x, Q, G, h, e, e)

        return F.log_softmax(x)
```

# Caveats

+ Make sure that your QP layer is always feasible.
  Otherwise it will become ill-defined.
  One way to do this is by selecting some <img src="https://rawgit.com/locuslab/qpth/master/svgs/d1a81d9dc6dd30e43ba27c5490a34a32.svg?invert_in_darkmode" align=middle width=14.519505pt height=13.38744pt/> and <img src="https://rawgit.com/locuslab/qpth/master/svgs/ac3148a5746b81298cb0c456b661f197.svg?invert_in_darkmode" align=middle width=14.58039pt height=13.38744pt/>
  and then setting <img src="https://rawgit.com/locuslab/qpth/master/svgs/bb48f07b397fe59eeedf7d02b3fa02bd.svg?invert_in_darkmode" align=middle width=94.003965pt height=22.06347pt/> and <img src="https://rawgit.com/locuslab/qpth/master/svgs/db45b41191573128ea7724a5d82935d3.svg?invert_in_darkmode" align=middle width=55.820655pt height=22.06347pt/>.
+ If your convergence seems instable, the solver may
  not be exactly solving them. Oftentimes, using doubles
  instead of floats will help the solver better approximate
  the solution.
+ See the "Limitation of the method" portion of our paper
  for some more notes.

# Acknowledgments

+ The rapid development of this work would not have been possible without
  the immense amount of help from the [PyTorch](https://pytorch.org) team,
  particularly [Soumith Chintala](http://soumith.ch/) and
  [Adam Paszke](https://github.com/apaszke).
+ The inline LaTeX in this README was created with
  [leegao/readme2tex](https://github.com/leegao/readme2tex)
  with [this script](https://github.com/locuslab/qpth/blob/master/make-readme.sh)
  in our repo.

# Citations

If you find this repository helpful in your publications,
please consider citing our paper.

```
@article{amos2017optnet,
  title={OptNet: Differentiable Optimization as a Layer in Neural Networks},
  author={Brandon Amos and J. Zico Kolter},
  journal={arXiv preprint arXiv:1703.00443},
  year={2017}
}
```

# Licensing

Unless otherwise stated, the source code is copyright
Carnegie Mellon University and licensed under the
[Apache 2.0 License](./LICENSE).

---

# Appendix

These sections are copied here from our paper for convenience.
See the paper for full references.

## How the forward pass works.

Deep networks are typically trained in mini-batches to take advantage
of efficient data-parallel GPU operations.
Without mini-batching on the GPU, many modern deep learning
architectures become intractable for all practical purposes.
However, today's state-of-the-art QP solvers like Gurobi and CPLEX
do not have the capability of solving multiple optimization
problems on the GPU in parallel across the entire minibatch.
This makes larger OptNet layers become quickly intractable
compared to a fully-connected layer with the same number of parameters.

To overcome this performance bottleneck in our quadratic program layers,
we have implemented a GPU-based primal-dual interior point
method (PDIPM) based on [mattingley2012cvxgen]
that solves a batch of quadratic programs, and which provides the necessary
gradients needed to train these in an end-to-end fashion.

Following the method of [mattingley2012cvxgen],
our solver introduces slack variables on the inequality constraints
and iteratively minimizes the residuals from the KKT conditions
over the primal and dual variables.
Each iteration computes the affine scaling directions by solving
<p align="center"><img src="https://rawgit.com/locuslab/qpth/master/svgs/ef2e0618a4200a63a14427f9339a7a26.svg?invert_in_darkmode" align=middle width=303.20565pt height=79.20066pt/></p>
where
<p align="center"><img src="https://rawgit.com/locuslab/qpth/master/svgs/4625b85f7d226e3e0c574c0becd42589.svg?invert_in_darkmode" align=middle width=222.2715pt height=78.91455pt/></p>
then centering-plus-corrector directions by solving
<p align="center"><img src="https://rawgit.com/locuslab/qpth/master/svgs/769ac47e86a3e41216843bff259a6900.svg?invert_in_darkmode" align=middle width=278.65695pt height=78.904815pt/></p>
where <img src="https://rawgit.com/locuslab/qpth/master/svgs/07617f9d8fe48b4a7b3f523d6730eef0.svg?invert_in_darkmode" align=middle width=10.227195pt height=13.38744pt/> is the duality gap and <img src="https://rawgit.com/locuslab/qpth/master/svgs/4003fd05114d4ca5530b1a98bf565fca.svg?invert_in_darkmode" align=middle width=40.441995pt height=20.41941pt/>.
Each variable <img src="https://rawgit.com/locuslab/qpth/master/svgs/6c4adbc36120d62b98deef2a20d5d303.svg?invert_in_darkmode" align=middle width=8.880135pt height=13.38744pt/> is updated with
<img src="https://rawgit.com/locuslab/qpth/master/svgs/ef02e65936f4e341b952a3a4d95ffd6c.svg?invert_in_darkmode" align=middle width=136.880535pt height=27.14481pt/>
using an appropriate step size.

We solve these iterations for every example in our
minibatch by solving a symmetrized version
of these linear systems with
<p align="center"><img src="https://rawgit.com/locuslab/qpth/master/svgs/eab9b4ace0c7c3f5e43a4fa0dd3c25a2.svg?invert_in_darkmode" align=middle width=249.47835pt height=78.91455pt/></p>
where
<img src="https://rawgit.com/locuslab/qpth/master/svgs/269399d5506c3a5c963c2343104d6303.svg?invert_in_darkmode" align=middle width=52.6878pt height=23.88969pt/> is the only portion of <img src="https://rawgit.com/locuslab/qpth/master/svgs/7d96c09e758450152d9c5e8cc5f9bb88.svg?invert_in_darkmode" align=middle width=37.19694pt height=21.69783pt/>
that changes between iterations.
We analytically decompose these systems into smaller
symmetric systems and pre-factorize portions of them
that don't change (i.e. that don't involve <img src="https://rawgit.com/locuslab/qpth/master/svgs/269399d5506c3a5c963c2343104d6303.svg?invert_in_darkmode" align=middle width=52.6878pt height=23.88969pt/>
between iterations.

## QP layers and backpropagation

Training deep architectures, however, requires that we not just have a forward
pass in our network but also a backward pass. This requires that we compute the
derivative of the solution to the QP with respect to its input parameters.
Although the previous papers mentioned above have considered similar argmin
differentiation techniques [gould2016differentiating], to the best of
our knowledge this is the
first case of a general formulation for argmin differentiation in the presence
of exact equality and inequality constraints.  To obtain these derivatives, we
we differentiate the KKT conditions (sufficient and necessary condition for
optimality) of a QP at a solution to the problem,  using techniques
from matrix differential calculus [magnus1988matrix].
Our analysis here can be extended to
more general convex optimization problems.

The Lagrangian of a QP is given by
<p align="center"><img src="https://rawgit.com/locuslab/qpth/master/svgs/b7ed8ee1f053c9f097ec678ddfaf66d7.svg?invert_in_darkmode" align=middle width=389.9643pt height=32.9901pt/></p>
where <img src="https://rawgit.com/locuslab/qpth/master/svgs/b49211c7e49541e500c32b4d56d354dc.svg?invert_in_darkmode" align=middle width=9.488985pt height=13.38744pt/> are the dual variables on the equality constraints
and <img src="https://rawgit.com/locuslab/qpth/master/svgs/946337dbecc318a448ec388a54c058a2.svg?invert_in_darkmode" align=middle width=40.048305pt height=22.06347pt/> are the dual variables on the inequality constraints.
The KKT conditions for stationarity, primal feasibility,
and complementary slackness are
<p align="center"><img src="https://rawgit.com/locuslab/qpth/master/svgs/ad6cdd9c5d555650c8a23c30f5fd8311.svg?invert_in_darkmode" align=middle width=212.0019pt height=68.074875pt/></p>
where <img src="https://rawgit.com/locuslab/qpth/master/svgs/018331bd713cc600188bd0b9e95ffab2.svg?invert_in_darkmode" align=middle width=31.740225pt height=23.88969pt/> creates a diagonal matrix from a vector. Taking the
differentials of these conditions gives the equations
<p align="center"><img src="https://rawgit.com/locuslab/qpth/master/svgs/3147025e7cf26e68bd680bcc642eb1e9.svg?invert_in_darkmode" align=middle width=335.63145pt height=95.21853pt/></p>
or written more compactly in matrix form
<p align="center"><img src="https://rawgit.com/locuslab/qpth/master/svgs/d555db93ac76836b5bd40d75b0d810a2.svg?invert_in_darkmode" align=middle width=279.75585pt height=124.96209pt/></p>
Using these equations, we can form the Jacobians of <img src="https://rawgit.com/locuslab/qpth/master/svgs/2645329b9d424bc5471c3e8db0a524e5.svg?invert_in_darkmode" align=middle width=15.42519pt height=21.87075pt/> (or
<img src="https://rawgit.com/locuslab/qpth/master/svgs/e8534c64081cb64af2d5ec9fb8b77a31.svg?invert_in_darkmode" align=middle width=16.64652pt height=22.06347pt/> and <img src="https://rawgit.com/locuslab/qpth/master/svgs/e3493376e9ac4374b09150aa58626b09.svg?invert_in_darkmode" align=middle width=16.22412pt height=21.87075pt/>, though we don't consider this case here), with
respect to any of the data parameters.  For example, if we wished to compute the
Jacobian <img src="https://rawgit.com/locuslab/qpth/master/svgs/5a746e9c137bf8f9724b455d0eaebbe1.svg?invert_in_darkmode" align=middle width=85.668pt height=29.74686pt/>, we
would simply substitute <img src="https://rawgit.com/locuslab/qpth/master/svgs/d10d846740fc6563b96ace37a3cf67b7.svg?invert_in_darkmode" align=middle width=46.30395pt height=22.06347pt/> (and set all other differential terms in
the right hand side to zero), solve the equation, and the resulting value of
<img src="https://rawgit.com/locuslab/qpth/master/svgs/55f4fdead12555ba180e515cdfb16322.svg?invert_in_darkmode" align=middle width=17.1831pt height=22.06347pt/> would be the desired Jacobian.

In the backpropagation algorithm, however, we never want to explicitly form the
actual Jacobian matrices, but rather want to form the left matrix-vector product
with some previous backward pass vector <img src="https://rawgit.com/locuslab/qpth/master/svgs/a9df0575617e7afad7025f678b0dcee7.svg?invert_in_darkmode" align=middle width=63.729105pt height=28.15857pt/>, i.e., <img src="https://rawgit.com/locuslab/qpth/master/svgs/b0aa7baa02d182727ddaa3bc132b9b93.svg?invert_in_darkmode" align=middle width=46.96461pt height=29.74686pt/>.   We can do this efficiently by noting the
solution for the <img src="https://rawgit.com/locuslab/qpth/master/svgs/361126cf129ba5757682f90310b623a9.svg?invert_in_darkmode" align=middle width=80.322495pt height=23.88969pt/> involves multiplying the *inverse* of the left-hand-side matrix above by some right hand
side.  Thus, if multiply the backward pass vector by the transpose of the
differential matrix
<p align="center"><img src="https://rawgit.com/locuslab/qpth/master/svgs/f44256ffcda2365c281c74d2590e6446.svg?invert_in_darkmode" align=middle width=329.7756pt height=62.540115pt/></p>
then the relevant gradients with respect to all the QP parameters can be given by
<p align="center"><img src="https://rawgit.com/locuslab/qpth/master/svgs/8cec70675f1499b79fc8c2e14b8f90d5.svg?invert_in_darkmode" align=middle width=337.7517pt height=164.56341pt/></p>
where as in standard backpropagation, all these terms are at most the size of
the parameter matrices.

## Efficiently computing gradients.

A key point of the particular form of primal-dual interior point method that we
employ is that it is possible to compute the backward pass gradients "for
free" after solving the original QP, without an additional matrix factorization
or solve.  Specifically, at each iteration in the primal-dual interior point, we
are computing an LU decomposition of the matrix <img src="https://rawgit.com/locuslab/qpth/master/svgs/e9092102e532684824c1f06bb6eeb88e.svg?invert_in_darkmode" align=middle width=37.19694pt height=21.69783pt/>. (We
actually perform an LU decomposition of a certain subset of the matrix formed
by eliminating variables to create only a <img src="https://rawgit.com/locuslab/qpth/master/svgs/69ab598334e622c64ab47719ce55379d.svg?invert_in_darkmode" align=middle width=36.954555pt height=18.41037pt/> matrix (the number of
inequality constraints) that needs to be factor during each iteration of the
primal-dual algorithm, and one <img src="https://rawgit.com/locuslab/qpth/master/svgs/49479e63f34aff060b55405e0b8b5b69.svg?invert_in_darkmode" align=middle width=49.279725pt height=18.41037pt/> and one <img src="https://rawgit.com/locuslab/qpth/master/svgs/3add1221abfa79cb14021bc2dacd5725.svg?invert_in_darkmode" align=middle width=40.147305pt height=18.41037pt/> matrix once at
the start of the primal-dual algorithm, though we omit the detail here.  We also
use an LU decomposition as this routine is provided in batch form by CUBLAS, but
could potentially use a (faster) Cholesky factorization if and when the
appropriate functionality is added to CUBLAS).)
This matrix is essentially a
symmetrized version of the matrix needed for computing the backpropagated
gradients, and we can similarly compute the <img src="https://rawgit.com/locuslab/qpth/master/svgs/d4dd40bb6606c0b78711670fb54aa3d8.svg?invert_in_darkmode" align=middle width=38.59317pt height=22.06347pt/> terms by solving
the linear system
<p align="center"><img src="https://rawgit.com/locuslab/qpth/master/svgs/9b08078786ae47ff0904d82530e2036c.svg?invert_in_darkmode" align=middle width=210.64725pt height=90.667335pt/></p>
where <img src="https://rawgit.com/locuslab/qpth/master/svgs/d4dd78836afa1d3a112b83652efa4110.svg?invert_in_darkmode" align=middle width=99.76527pt height=29.865pt/> for <img src="https://rawgit.com/locuslab/qpth/master/svgs/f620a4407b2e1112a8e05a65263701d4.svg?invert_in_darkmode" align=middle width=16.675065pt height=22.06347pt/> as
defined earlier. Thus, all the backward pass gradients can be computed
using the factored KKT matrix at the solution.  Crucially, because the
bottleneck of solving this linear system is computing the factorization of the
KKT matrix (cubic time as opposed to the quadratic time for solving via
backsubstitution once the factorization is computed), the additional time
requirements for computing all the necessary gradients in the backward pass is
virtually nonexistent compared with the time of computing the solution.  To the
best of our knowledge, this is the first time that this fact has been exploited
in the context of learning end-to-end systems.
