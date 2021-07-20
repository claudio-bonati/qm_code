#!/usr/bin/env python3

import numpy as np
import sch_solver_dense as ssd
import sch_solver_sparse as sss
import sys

__all__ = ["richardson_extrap"]

#***************************
#library functions and classes

def richardson_extrap(solver, size):
  """Richardson extrapolation of the solver 

  solver must have a method solve(size) to apply this function
  """
  if(solver.approxorder==None):
    print("ERROR: Richardson extrapolation can not be applied to the selected solver")
    sys.exit(1)

  ris1=solver.solve(size)
  numeigs=solver.numeigs 

  ris2=solver.solve(2*size)

  aux=np.power(2.0,solver.approxorder)

  return (aux*ris2[:numeigs]-ris1)/(aux-1.0)



#***************************
# unit testing



def _test_dense(solmethod, pot, xmin, xmax, N, numeigs):
  print("Some results")
  solver=solmethod(pot, xmin, xmax)

  risA=solver.solve(N)
  risB=solver.solve(2*N)
  risC=solver.solve(4*N)
  risD=richardson_extrap(solver, 2*N)

  print(' i    size={:>2d}         '.format(N), end='')
  print('size={:>2d}         size={:>2d}         '.format(2*N, 4*N), end='')
  print('size={:>2d} Rich.'.format(2*N))
  for i in range(numeigs):
     print('{:>2d} {:>15.10f} {:>15.10f} {:>15.10f} {:>15.10f}'.format(i, risA[i], risB[i], risC[i], risD[i]))
  print()



def _test_sparse(solmethod, pot, xmin, xmax, N, numeigs):
  print("Some results")
  solver=solmethod(pot, xmin, xmax, numeigs)

  risA=solver.solve(N)
  risB=solver.solve(2*N)
  risC=solver.solve(4*N)
  risD=richardson_extrap(solver, 2*N)

  print(' i    size={:>2d}         '.format(N), end='')
  print('size={:>2d}         size={:>2d}         '.format(2*N, 4*N), end='')
  print('size={:>2d} Rich.'.format(2*N))
  for i in range(numeigs):
     print('{:>2d} {:>15.10f} {:>15.10f} {:>15.10f} {:>15.10f}'.format(i, risA[i], risB[i], risC[i], risD[i]))
  print()



if __name__=="__main__":
  
  print("**********************")
  print("UNIT TESTING")
  print()

  print("TEST 1: harmonic oscillator on [-10:10] (hbar=m=omega=1)")
  print()

  def pot_h(x):
    return x*x/2.0
  xmin=-10
  xmax=10
  N=20
  numeigs=5

  print("****")
  print("FDM Dense three point discretizion with Dirichlet b.c.")
  print()
  _test_dense(ssd.FDM_Dense_3pD, pot_h, xmin, xmax, N, numeigs)
  print("****")
  print("FDM Sparse three point discretizion with Dirichlet b.c.")
  print()
  _test_sparse(sss.FDM_Sparse_3pD, pot_h, xmin, xmax, N, numeigs)

  print("****")
  print("FDM Dense five point discretizion with Dirichlet b.c.")
  print()
  _test_dense(ssd.FDM_Dense_5pD, pot_h, xmin, xmax, N, numeigs)
  print("****")
  print("FDM Sparse five point discretizion with Dirichlet b.c.")
  print()
  _test_sparse(sss.FDM_Sparse_5pD, pot_h, xmin, xmax, N, numeigs)

  print("**********************")

