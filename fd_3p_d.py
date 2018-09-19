#!/usr/bin/env python3

import numpy as np
from scipy import linalg
import sys

__all__ = ["fd_3p_d_create_matrix", "fd_3p_d_solver", "fd_3p_d_solver_extrap"]

#***************************
#library functions

def fd_3p_d_create_matrix(potential, xmin, xmax, N):
  """Create the matrix for the finite difference 3 points discretization
  with Dirichlet boundary conditions.

  potential = potential energy to be used
  the problem is defined on the interval [xmin, xmax]
  N = size of the matrix to be used

  return a NxN numpy array
  """

  if not isinstance(N, int):
    print("ERROR: N has to be an integer!")
    sys.exit(1)

  if N<2:
    print("ERROR: N need to be at least 2!")
    sys.exit(1)

  if xmin>xmax:
    print("ERROR: xmax has to be larger than xmin!")
    sys.exit(1)


  M=np.zeros((N, N), dtype=np.float)

  step=(xmax-xmin)/N
  for i in range(0, N, 1):
    M[i][i]=2.0+step*step*potential(xmin+i*step)

  for i in range(0, N-1, 1):
    M[i][i+1]=1.0
    M[i+1][i]=1.0

  return M 


def fd_3p_d_solver(potential, xmin, xmax, N):
  """Find the eigenvalues of the Schrodinger equation
  -\psi''(x)+V(x)\psi(x)=E\psi(x)
  using a finite difference 3 point discretization with
  Dirichlet boundary conditions.

  potential = V(x) 
  the problem is defined on the interval [xmin, xmax]
  N = size of the matrix to be used

  return a N eigenvalues as a sorted numpy array
  """

  M=fd_3p_d_create_matrix(potential, xmin, xmax, N)

  eigval=linalg.eigvals(M)

  step=(xmax-xmin)/N
  eigval/=(step*step)

  im_part=np.imag(eigval)
  re_part=np.real(eigval)

  aux=np.max(np.abs(im_part))
  if aux>1.0e-12:
    print("WARNING: imaginary part large {:g}".format(aux))
    print("")

  ris=np.sort(re_part)  

  return ris


def fd_3p_d_solver_extrap(potential, xmin, xmax, N):
  """Find the eigenvalues of the Schrodinger equation
  -\psi''(x)+V(x)\psi(x)=E\psi(x)
  using a finite difference 3 point discretization with
  Dirichlet boundary conditions and Richardson extrapolation

  potential = V(x) 
  the problem is defined on the interval [xmin, xmax]
  N = size initial of the matrix to be used

  return a N eigenvalues as a sorted numpy array
  """

  ris1=fd_3p_d_solver(potential, xmin, xmax, N)
  ris2=fd_3p_d_solver(potential, xmin, xmax, 2*N)

  aux=ris2[:N]
  
  ris=(4.0*aux-ris1)/3.0

  return ris


#***************************
# unit testing

if __name__=="__main__":
  
  print("**********************")
  print("UNIT TESTING")
  print()

  # with m=1/2 this is an harmonic oscillator with \omega=2
  # and eigenvalues E_n=2n+1
  def pot_h(x):
    return x*x

  xmin=-20
  xmax=20

  N=50

  risA=fd_3p_d_solver(pot_h, xmin, xmax, N)
  risB=fd_3p_d_solver(pot_h, xmin, xmax, 2*N)
  risC=fd_3p_d_solver(pot_h, xmin, xmax, 3*N)
  risD=fd_3p_d_solver(pot_h, xmin, xmax, 4*N)

  risB_extrap=fd_3p_d_solver_extrap(pot_h, xmin, xmax, 2*N)

  print('Harmonic oscillator on [{:+.2f}, {:+.2f}]'.format(xmin, xmax))
  print('3 points finite difference with Dirichlet boundary conditions, N={:d}'.format(N))
  print('')
  print('{:>2s} {:>15s} {:>15s} {:>15s} {:>15s} {:>15s} {:>15s}'.format("n", "N", "2N", "3N", "4N", "2N extr", "exact"))
  for i in range(0, 20, 2):
     print('{:>2d} {:>15.10f} {:>15.10f} {:>15.10f} {:>15.10f} {:>15.10f} {:>15.10f}'.format(i, risA[i], risB[i], risC[i], risD[i], risB_extrap[i], 2*i+1))

  print('')
  print('')

  ## Paine potential
  def pot_p(x):
    return 1/(x+0.1)/(x+0.1)

  def exact_p(x):
    if x==0:
      return 1.5198658211
    if x==1:
      return 4.9433098221
    if x==2:
      return 10.284662645
    if x==3:
      return 17.559957746
    if x==4:
      return 26.782863158
    if x==5:
      return 37.964425862
    if x==6:
      return 51.113357757
    if x==7:
      return 66.236447704
    if x==8:
      return 83.338962374
    if x==9:
      return 102.42498840
    if x==10:
      return 123.49770680
    if x==11:
      return 146.55960608
    if x==12:
      return 171.61264485
    if x==13: 
      return 198.65837500
    if x==14:
      return 227.69803474
    if x==15:
      return 258.73261893
    if x==16:
      return 291.76293246
    if x==17:
      return 326.78963096
    if x==18:
      return 363.81325194
    if x==19:
      return 402.83423888
    if x==20:
      return 443.85295984
    else:
      print("ERROR: the required value is not tabulated")
      sys.exit(1)

  xmin=0
  xmax=np.pi

  N=50

  risA=fd_3p_d_solver(pot_p, xmin, xmax, N)
  risB=fd_3p_d_solver(pot_p, xmin, xmax, 2*N)
  risC=fd_3p_d_solver(pot_p, xmin, xmax, 3*N)
  risD=fd_3p_d_solver(pot_p, xmin, xmax, 4*N)

  risB_extrap=fd_3p_d_solver_extrap(pot_p, xmin, xmax, 2*N)

  print('Paine problem (see Pryce "Numerical solution of Sturm-Liouville problems" App. A)')
  print('shooting using initial values from fd_3p_d_solver_extrap with N={:d}'.format(N))
  print('')
  print('{:>2s} {:>15s} {:>15s} {:>15s} {:>15s} {:>15s} {:>15s}'.format("n", "N", "2N", "3N", "4N", "2N extr", "exact"))
  for i in range(0, 20, 2):
     print('{:>2d} {:>15.10f} {:>15.10f} {:>15.10f} {:>15.10f} {:>15.10f} {:>15.10f}'.format(i, risA[i], risB[i], risC[i], risD[i], risB_extrap[i], exact_p(i)))

  print("**********************")
