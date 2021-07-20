#!/usr/bin/env python3

import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import sys

__all__ = ["SparseSolver", "FDM_Sparse_3pD", "FDM_Sparse_5pD"]

#***************************
#library functions and classes


class SparseSolver:
  """Class for solvers using sparse matrices
     (1d Schrodinger equation with \hbar=1 and m=1)
  """
  def __init__(self, V, xmin, xmax, numeigs):
    """Parameters for the solver
    V = potential
    xmin, xmax = interval to be used by the solver
    """
    if xmin>=xmax:
      print("ERROR: xmax has to be larger than xmin")
      sys.exit(1)

    if not isinstance(numeigs, int):
      print("ERROR: numerigs has to be an integer")
      sys.exit(1)

    self.V=V
    self.xmin=xmin
    self.xmax=xmax
    self.numeigs=numeigs

    self.size=None
    self.M=None   

  def setup(self):
    self.solutionmethod=None
    self.approxorder=None

  def solve(self, size):
    """Return self.numerig eigenvalues of the problem 
       defined on a grid of extent 'size'
    """
    if not isinstance(size, int):
      print("ERROR: size has to be an integer")
      sys.exit(1)
    self.size=size

    self.setup()

    if self.solutionmethod==None:
      print("ERROR: method 'solve' can only be called by sublcass solver instances of SparceSolver")
      sys.exit(1)

    eigvals=linalg.eigs(self.M, k=self.numeigs, which='SM', return_eigenvectors=False)

    im_part=np.imag(eigvals)
    re_part=np.real(eigvals)

    aux=np.max(np.abs(im_part))
    if aux > 1.0e-12:
      print("WARNING: large imaginary part of some eigenvalue {:g}".format(aux))

    return np.sort(re_part)



class FDM_Sparse_3pD(SparseSolver):
  def setup(self):
    """Initialize the matrix for 3 point FDM with Dirichlet b.c.
    """
    self.solutionmethod="FDM_Sparse_3pD"
    self.approxorder=2

    self.step=(self.xmax-self.xmin)/float(self.size)
    invstep2=1.0/np.power(self.step,2)

    diag0 = np.ones(self.size, float)*invstep2 
    for i in range(self.size):
      diag0[i] += self.V(self.xmin+i*self.step)
    diag1=np.ones(self.size-1, float)*(-1.0/2.0*invstep2)

    self.M=sparse.diags([diag0, diag1, diag1], [0,-1,1])



class FDM_Sparse_5pD(SparseSolver):
  def setup(self):
    """Initialize the matrix for 5 point FDM with Dirichlet b.c.
    """
    self.solutionmethod="FDM_5pD"
    self.approxorder=4

    self.step=(self.xmax-self.xmin)/float(self.size)
    invstep2=1.0/np.power(self.step,2)

    diag0 = np.ones(self.size, float)*(5.0/4.0)*invstep2
    for i in range(self.size):
      diag0[i] += self.V(self.xmin+i*self.step)

    diag1=np.ones(self.size-1, float)*(-4.0/6.0*invstep2)
    diag2=np.ones(self.size-2, float)*(1.0/24.0*invstep2)

    self.M=sparse.diags([diag0, diag1, diag1, diag2, diag2], [0,-1,1,-2,2])



#***************************
# unit testing


def _test(solmethod, pot, xmin, xmax, N, numeigs, exact_first):
  print("Error scaling for fundamental level")
  print("size   error*size^approxorder")

  solver=solmethod(pot, xmin, xmax, numeigs)
  M=50
  for i in range(3):
    risA=solver.solve(M)
    test=(risA[1]-exact_first)*np.power(M, solver.approxorder)
    print(' {:>3d}  {:15.10f}'.format(M, test))
    M*=2
  print() 

  print("Some results")
  solver=solmethod(pot, xmin, xmax, numeigs)

  risA=solver.solve(N)
  risB=solver.solve(2*N)
  risC=solver.solve(4*N)

  print(' i    size={:>2d}         '.format(N), end='')
  print('size={:>2d}         size={:>2d}'.format(2*N, 4*N))
  for i in range(numeigs):
     print('{:>2d} {:>15.10f} {:>15.10f} {:>15.10f}'.format(i, risA[i], risB[i], risC[i]))
  print()



if __name__=="__main__":
  
  print("**********************")
  print("UNIT TESTING")
  print()

  print("TEST: harmonic oscillator on [-10:10] (hbar=m=omega=1)")
  print()

  def pot_h(x):
    return x*x/2.0
  xmin=-10
  xmax=10
  N=20
  numeigs=5

  print("****")
  print("FDM three point discretizion with Dirichlet b.c.")
  print()
  _test(FDM_Sparse_3pD, pot_h, xmin, xmax, N, numeigs, 1.5)
  print("****")
  print("FDM five point discretizion with Dirichlet b.c.")
  print()
  _test(FDM_Sparse_5pD, pot_h, xmin, xmax, N, numeigs, 1.5)

  print("**********************")

