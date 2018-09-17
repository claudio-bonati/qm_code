#!/usr/bin/env python3

import numpy as np
from scipy import fftpack  
from scipy import linalg
import sys

__all__ = ["fft_p_create_matrix", "fft_p_solver"]

#***************************
#library functions

def fft_p_create_matrix(potential, xmin, xmax, N, finegrid=65536):
  """Create the matrix for the fft method with periodic boundary conditions
  [1401.1178 and Bonati, Fagotti (unpublished)]

  potential = potential energy to be used
  the problem is defined on the interval [xmin, xmax]
  N = size of the matrix to be used
  finegrid=auxiliary very fine grid for the fft

  return a NxN numpy array
  """

  if not isinstance(N, int):
    print("ERROR: N has to be an integer!")
    sys.exit(1)

  if N<2:
    print("ERROR: N need to be at least 2!")
    sys.exit(1)

  if not N%2==0:
    N+=1

  if xmin>xmax:
    print("ERROR: xmax has to be larger than xmin!")
    sys.exit(1)

  step=(xmax-xmin)/finegrid

  potential_aux=np.empty(finegrid, dtype=np.complex)
  for i in range(finegrid):
    potential_aux[i]=potential(xmin+i*step)
  potential_fft=fftpack.ifft(potential_aux)

  M=np.zeros((N, N), dtype=np.complex)

  for i in range(N):
    M[i][i]= pow((i-N/2)*(2*np.pi/(xmax-xmin)), 2.0) + potential_fft[0]
    for j in range(i+1, N, 1):
      M[i][j]=potential_fft[abs(i-j)]
      M[j][i]=np.conjugate(potential_fft[abs(i-j)])

  return M 


def fft_p_solver(potential, xmin, xmax, N, finegrid=65536):
  """
  Find the eigenvalues of the Schrodinger equation
  -\psi''(x)+V(x)\psi(x)=E\psi(x)
  using the fft method with periodic boundary conditions
  [1401.1178 and Bonati, Fagotti (unpublished)]

  potential = potential energy to be used
  the problem is defined on the interval [xmin, xmax]
  N = size of the matrix to be used
  finegrid=auxiliary very fine grid for the fft

  return a N eigenvalues as a sorted numpy array
  """

  M=fft_p_create_matrix(potential, xmin, xmax, N, finegrid)

  eigval=linalg.eigvals(M)

  im_part=np.imag(eigval)
  re_part=np.real(eigval)

  aux=np.max(np.abs(im_part))
  if aux>1.0e-12:
    print("WARNING: imaginary part large {:g}".format(aux))
    print("")

  ris=np.sort(re_part)  

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

  risA=fft_p_solver(pot_h, xmin, xmax, N)
  risB=fft_p_solver(pot_h, xmin, xmax, 2*N)
  risC=fft_p_solver(pot_h, xmin, xmax, 3*N)
  risD=fft_p_solver(pot_h, xmin, xmax, 4*N)

  print('Harmonic oscillator on [{:+.2f}, {:+.2f}]'.format(xmin, xmax))
  print('fft method with periodic boundary conditions, N={:d}'.format(N))
  print('')
  print('{:>2s} {:>15s} {:>15s} {:>15s} {:>15s} {:>15s}'.format("n", "N", "2N", "3N", "4N", "exact"))
  for i in range(0, 20, 2):
     print('{:>2d} {:>15.10f} {:>15.10f} {:>15.10f} {:>15.10f} {:>15.10f}'.format(i, risA[i], risB[i], risC[i], risD[i], 2*i+1))

  print('')
  print('')

  def pot_zero(x):
    return 0

  xmin=0
  xmax=2*np.pi

  N=50

  risA=fft_p_solver(pot_zero, xmin, xmax, N)
  risB=fft_p_solver(pot_zero, xmin, xmax, 2*N)
  risC=fft_p_solver(pot_zero, xmin, xmax, 3*N)
  risD=fft_p_solver(pot_zero, xmin, xmax, 4*N)

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
  print('fft method with periodic boundary conditions, N={:d}'.format(N))
  print('')
  print('{:>2s} {:>15s} {:>15s} {:>15s} {:>15s} {:>15s}'.format("n", "N", "2N", "3N", "4N", "exact"))
  for i in range(0, 20, 2):
     print('{:>2d} {:>15.10f} {:>15.10f} {:>15.10f} {:>15.10f} {:>15.10f}'.format(i, risA[i], risB[i], risC[i], risD[i], exact_ris(i)))

  print("**********************")
