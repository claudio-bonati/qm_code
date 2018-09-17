#!/usr/bin/env python3

import numpy as np
from scipy import linalg
import sys

__all__ = ["fd_3p_p_create_matrix", "fd_3p_p_solver", "fd_3p_p_solver_extrap"]

#***************************
#library functions

def fd_3p_p_create_matrix(potential, xmin, xmax, N):
  """Create the matrix for the finite difference 3 points discretization
  with periodic boundary conditions.

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

  M[0][N-1]=1.0
  M[N-1][0]=1.0

  return M 


def fd_3p_p_solver(potential, xmin, xmax, N):
  """Find the eigenvalues of the Schrodinger equation
  -\psi''(x)+V(x)\psi(x)=E\psi(x)
  using a finite difference 3 point discretization with
  periodic boundary conditions.

  potential = V(x) 
  the problem is defined on the interval [xmin, xmax]
  N = size of the matrix to be used

  return a N eigenvalues as a sorted numpy array
  """

  M=fd_3p_p_create_matrix(potential, xmin, xmax, N)

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


def fd_3p_p_solver_extrap(potential, xmin, xmax, N):
  """Find the eigenvalues of the Schrodinger equation
  -\psi''(x)+V(x)\psi(x)=E\psi(x)
  using a finite difference 3 point discretization with
  periodic boundary conditions and Richardson extrapolation

  potential = V(x) 
  the problem is defined on the interval [xmin, xmax]
  N = size initial of the matrix to be used

  return a N eigenvalues as a sorted numpy array
  """

  ris1=fd_3p_p_solver(potential, xmin, xmax, N)
  ris2=fd_3p_p_solver(potential, xmin, xmax, 2*N)

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

  risA=fd_3p_p_solver(pot_h, xmin, xmax, N)
  risB=fd_3p_p_solver(pot_h, xmin, xmax, 2*N)
  risC=fd_3p_p_solver(pot_h, xmin, xmax, 3*N)
  risD=fd_3p_p_solver(pot_h, xmin, xmax, 4*N)

  risB_extrap=fd_3p_p_solver_extrap(pot_h, xmin, xmax, 2*N)

  print('Harmonic oscillator on [{:+.2f}, {:+.2f}]'.format(xmin, xmax))
  print('3 points finite difference with periodic boundary conditions, N={:d}'.format(N))
  print('')
  print('{:>2s} {:>15s} {:>15s} {:>15s} {:>15s} {:>15s} {:>15s}'.format("n", "N", "2N", "3N", "4N", "2N extr", "exact"))
  for i in range(0, 20, 2):
     print('{:>2d} {:>15.10f} {:>15.10f} {:>15.10f} {:>15.10f} {:>15.10f} {:>15.10f}'.format(i, risA[i], risB[i], risC[i], risD[i], risB_extrap[i], 2*i+1))

  print('')
  print('')



  def pot_zero(x):
    return 0

  xmin=0
  xmax=2*np.pi

  N=50

  risA=fd_3p_p_solver(pot_zero, xmin, xmax, N)
  risB=fd_3p_p_solver(pot_zero, xmin, xmax, 2*N)
  risC=fd_3p_p_solver(pot_zero, xmin, xmax, 3*N)
  risD=fd_3p_p_solver(pot_zero, xmin, xmax, 4*N)

  risB_extrap=fd_3p_p_solver_extrap(pot_zero, xmin, xmax, 2*N)

  def exact_ris(n):
   if not isinstance(n, int):
     print("ERROR: n has to be an integer!")
     sys.exit(1)
   if N<0:
     print("ERROR: n has to be positive!")
     sys.exit(1)
   if n==0:
     return 0
   else:
     aux=int(n/2)
     return aux*aux

  print('Zero potential on [{:+.2f}, {:+.6f}]'.format(xmin, xmax))
  print('3 points finite difference with periodic boundary conditions, N={:d}'.format(N))
  print('')
  print('{:>2s} {:>15s} {:>15s} {:>15s} {:>15s} {:>15s} {:>15s}'.format("n", "N", "2N", "3N", "4N", "2N extr", "exact"))
  for i in range(0, 20, 2):
     print('{:>2d} {:>15.10f} {:>15.10f} {:>15.10f} {:>15.10f} {:>15.10f} {:>15.10f}'.format(i, risA[i], risB[i], risC[i], risD[i], risB_extrap[i], exact_ris(i)))

  print("**********************")
