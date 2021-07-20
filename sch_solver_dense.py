#!/usr/bin/env python3

import numpy as np
from scipy import fftpack  
from scipy import linalg
import sys

__all__ = ["DenseSolver", "FDM_Dense_3pD", "FDM_Dense_3pP", "FDM_Dense_5pD", "FDM_Dense_5pP", "FFTPeriodic"]

#***************************
#library functions and classes

class DenseSolver:
  """Class for solvers using dense matrices
     (1d Schrodinger equation with \hbar=1 and m=1)
  """
  def __init__(self, V, xmin, xmax):
    """Parameters for the solver
    V = potential
    xmin, xmax = interval to be used by the solver
    """
    if xmin>=xmax:
      print("ERROR: xmax has to be larger than xmin")
      sys.exit(1)

    self.V=V
    self.xmin=xmin
    self.xmax=xmax

    self.size=None
    self.M=None   

  def setup(self):
    self.solutionmethod=None
    self.approxorder=None

  def solve(self, size):
    """Return a sorted list of size eigenvalues
    """
    if not isinstance(size, int):
      print("ERROR: size has to be an integer")
      sys.exit(1)
    self.size=size
    self.numeigs=size 

    self.setup()

    if self.solutionmethod==None:
      print("ERROR: method 'solve' can only be called by sublcass solver instances of DenseSolver")
      sys.exit(1)

    eigvals=linalg.eigvals(self.M)

    im_part=np.imag(eigvals)
    re_part=np.real(eigvals)

    aux=np.max(np.abs(im_part))
    if aux > 1.0e-12:
      print("WARNING: large imaginary part of some eigenvalue {:g}".format(aux))

    return np.sort(re_part)



class FDM_Dense_3pD(DenseSolver):
  def setup(self):
    """Initialize the matrix for 3 point FDM with Dirichlet b.c.
    """
    self.solutionmethod="FDM_3pD"
    self.approxorder=2

    self.step=(self.xmax-self.xmin)/float(self.size)
    invstep2=1.0/np.power(self.step,2)

    self.M=np.zeros((self.size, self.size), float)
    for i in range(self.size):
      self.M[i, i] = self.V(self.xmin+i*self.step)
      self.M[i, i] += invstep2

    for i in range(self.size-1):
      self.M[i, i+1] = -1.0/2.0*invstep2
      self.M[i+1, i] = -1.0/2.0*invstep2



class FDM_Dense_3pP(DenseSolver):
  def setup(self):
    """Initialize the matrix for 3 point FDM with periodic b.c.
    """
    self.solutionmethod="FDM_3pP"
    self.approxorder=2

    self.step=(self.xmax-self.xmin)/float(self.size)
    invstep2=1.0/np.power(self.step,2)

    self.M=np.zeros((self.size, self.size), float)
    for i in range(self.size):
      self.M[i, i] = self.V(self.xmin+i*self.step)
      self.M[i, i] += invstep2

      j=(i+1) % self.size
      self.M[i, j] = -1.0/2.0*invstep2
      self.M[j, i] = -1.0/2.0*invstep2



class FDM_Dense_5pD(DenseSolver):
  def setup(self):
    """Initialize the matrix for 5 point FDM with Dirichlet b.c.
    """
    self.solutionmethod="FDM_5pD"
    self.approxorder=4

    self.step=(self.xmax-self.xmin)/float(self.size)
    invstep2=1.0/np.power(self.step,2)

    self.M=np.zeros((self.size, self.size), float)
    for i in range(self.size):
      self.M[i, i] = self.V(self.xmin+i*self.step)
      self.M[i, i] += 5.0/4.0*invstep2
    
    for i in range(self.size-1):
      self.M[i, i+1] = -4.0/6.0*invstep2 
      self.M[i+1, i] = -4.0/6.0*invstep2

    for i in range(self.size-2):
      self.M[i, i+2] = 1.0/24.0*invstep2 
      self.M[i+2, i] = 1.0/24.0*invstep2



class FDM_Dense_5pP(DenseSolver):
  def setup(self):
    """Initialize the matrix for 5 point FDM with periodic b.c.
    """
    self.solutionmethod="FDM_5pP"
    self.approxorder=4

    self.step=(self.xmax-self.xmin)/float(self.size)
    invstep2=1.0/np.power(self.step,2)

    self.M=np.zeros((self.size, self.size), float)

    for i in range(self.size):
      self.M[i, i] = self.V(self.xmin+i*self.step)
      self.M[i, i] += 5.0/4.0*invstep2

      j=(i+1) % self.size
      self.M[i, j] = -4.0/6.0*invstep2 
      self.M[j, i] = -4.0/6.0*invstep2

      k=(i+2) % self.size
      self.M[i, k] = 1.0/24.0*invstep2 
      self.M[k, i] = 1.0/24.0*invstep2



class FFTPeriodic(DenseSolver):
  def setup(self):
    """Initialize the matrix for the fft method with periodic b.c.
    [arXiv:1401.1178 and Bonati, Fagotti (unpublished)]
    """
    self.solutionmethod="FFTPeriodic"
    self.approxorder=None

    if not self.size%2==0:
      self.size+=1

    if xmin>xmax:
      print("ERROR: xmax has to be larger than xmin!")
      sys.exit(1)

    finegrid=65536 #=2^{16} any large power of 2 would work
    if(100*self.size>finegrid):
      tmp=int(np.log2(self.size))
      finegrid=np.power(2, tmp+5)
    finestep=(self.xmax-self.xmin)/finegrid

    pot_aux=np.empty(finegrid, dtype=complex)
    for i in range(finegrid):
      pot_aux[i]=self.V(self.xmin+i*finestep)

    pot_fft=fftpack.ifft(pot_aux)

    self.M=np.zeros((self.size, self.size), dtype=np.complex)
    for i in range(self.size):
      self.M[i, i]= (1.0/2.0)*pow((i-self.size/2.0)*(2.0*np.pi/(self.xmax-self.xmin)), 2.0) + pot_fft[0]
      for j in range(i+1, self.size, 1):
        self.M[i, j]=pot_fft[abs(i-j)]
        self.M[j, i]=np.conjugate(pot_fft[abs(i-j)])




#***************************
# unit testing


def _test1(solmethod, pot, xmin, xmax, N, exact_first):
  print("Error scaling for fundamental level")
  print("size   error*size^approxorder")

  solver=solmethod(pot, xmin, xmax)

  M=50
  for i in range(3):
    risA=solver.solve(M)
    test=(risA[1]-exact_first)*np.power(M, solver.approxorder)
    print(' {:>3d} {:15.10f}'.format(M, test))
    M*=2
  print() 

  print("Some results")
  solver=solmethod(pot, xmin, xmax)

  risA=solver.solve(N)
  risB=solver.solve(2*N)
  risC=solver.solve(4*N)

  print(' i     size={:>2d}         '.format(N), end='')
  print('size={:>2d}         size={:>2d}'.format(2*N, 4*N))
  for i in range(5):
     print('{:>2d} {:>15.10f} {:>15.10f} {:>15.10f}'.format(i, risA[i], risB[i], risC[i]))
  print()



def _test2(solmethod, pot, xmin, xmax, N):
  print("Some results")
  solver=solmethod(pot, xmin, xmax)

  risA=solver.solve(N)
  risB=solver.solve(2*N)
  risC=solver.solve(4*N)

  print(' i    size={:>2d}         '.format(N), end='')
  print('size={:>2d}        size={:>2d}'.format(2*N, 4*N))
  for i in range(5):
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

  print("****")
  print("FDM three point discretizion with Dirichlet b.c.")
  print()
  _test1(FDM_Dense_3pD, pot_h, xmin, xmax, N, 1.5)
  print("****")
  print("FDM five point discretizion with Dirichlet b.c.")
  print()
  _test1(FDM_Dense_5pD, pot_h, xmin, xmax, N, 1.5)
  print("****")
  print("FFT with periodic b.c.")
  print()
  _test2(FFTPeriodic, pot_h, xmin, xmax, N)
  print("****")
  print()



  print("TEST: vanishing potential on [0,pi] (hbar=m=omega=1)")
  print()

  def pot_h(x):
    return 0.0
  xmin=0
  xmax=np.pi
  N=10

  print("****")
  print("FDM three point discretizion with periodic b.c.")
  print()
  _test1(FDM_Dense_3pP, pot_h, xmin, xmax, N, 2.0)
  print("****")
  print("FDM five point discretizion with periodic b.c.")
  print()
  _test1(FDM_Dense_5pP, pot_h, xmin, xmax, N, 2.0)

  print("**********************")

