#!/usr/bin/env python3

import numpy as np
from scipy import optimize
import sys

__all__ = ["ShootingSolverDirichlet", "Shoot_3p", "Shoot_Numerov"]

#***************************
#library functions and classes

class ShootingSolverDirichlet:
  """Class for solvers using shooting and Dirichlet b.c.
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

  def solveinitialproblem(self, E, y0, y0prime):
    self.solutionmethod=None
    self.approxorder=None

  def solve(self, size, E0seed):
    """Return the refined estimate of the energy level E0seed
    obtained by using a mesh of extent size
    """
    if not isinstance(size, int):
      print("ERROR: size has to be an integer")
      sys.exit(1)
    self.size=size
    self.numeigs=1

    def f(x):
      final = self.solveinitialproblem(x, 0.0, 1.0e-8)
      return final

    ris=optimize.newton(f, x0=E0seed, tol=1.0e-8)
     
    return ris

  def __call__(self, size, E0seed):
    return self.solve(size, E0seed)
 
  def solve_up_to_toll(self, E0seed, tollerance, sizeseed=1000):
    """Return a refined estimate of the energy level E0seed
    accurate to 'tollerance'
    """
    size=sizeseed

    ris0=self.solve(size, E0seed)
    
    delta=1.0
    while delta>tollerance:
      size*=2
      ris1=self.solve(size, ris0)
      delta=np.abs(ris1-ris0)
      ris0=ris1

    #if the algorthms end size/2 is enough
    self.size=int(size/2)
  
    return ris0



class Shoot_3p(ShootingSolverDirichlet):
  def solveinitialproblem(self, E, y0, y0prime):
    """Solve the initial problem with energy E and 
    y(xmin)=y0
    y'(xmin)=y0
    using 3 point discretization.
    """
    self.solutionmethod="Shoot_3p"
    self.approxorder=2

    step=(self.xmax-self.xmin)/float(self.size)

    psi_im1=y0
    psi_i=y0+y0prime*step+(self.V(self.xmin)-E)*y0*pow(step,2)

    for i in range(1, self.size, 1):
      psi_ip1=2*psi_i - psi_im1 - 2*step*step*(E*psi_i-self.V(self.xmin + i*step)*psi_i)

      psi_im1=psi_i 
      psi_i=psi_ip1

    return psi_i 



class Shoot_Numerov(ShootingSolverDirichlet):
  def solveinitialproblem(self, E, y0, y0prime):
    """Solve the initial problem with energy E and 
    y(xmin)=y0
    y'(xmin)=y0
    using Numerov discretization.
    """
    self.solutionmethod="Shoot_Numerov"
    self.approxorder=4

    step=(self.xmax-self.xmin)/float(self.size)

    def v(x):
      return 2*(self.V(x)-E)

    psi_im1=y0
    xmin=self.xmin

    # see Quiroz Gonzalez and Thompson 
    # "Getting started with Numerov's method" COMPUTERS IN PHYSICS, VOL. 11, 514 (1997)
    # Eq. 15
    # DOI: 10.1063/1.168593 
    num  = y0*(1-v(xmin+2*step)*pow(step,2)/24)
    num += step*y0prime*(1-v(xmin+2*step)*pow(step,2)/12) 
    num += pow(step,2)*7*y0*v(xmin)/24.0 -pow(step, 4)*v(xmin+2*step)*y0*v(xmin)/36.0
    den = 1 -v(xmin+step)*pow(step,2)/4 + v(xmin+step)*v(xmin+2*step)*pow(step, 4)/18.0
    psi_i=num/den

    # see e.g. ref. cit. Eq.7
    for i in range(1, self.size, 1):
      num=2*psi_i -psi_im1 +(10*v(xmin+i*step)*psi_i +v(xmin+(i-1)*step)*psi_im1)*pow(step,2)/12 
      den=1 -v(xmin+(i+1)*step)*pow(step,2)/12
      psi_ip1 =num/den 

      psi_im1=psi_i 
      psi_i=psi_ip1

    return psi_i 



#***************************
# unit testing


if __name__=="__main__":
  
  print("**********************")
  print("UNIT TESTING")
  print()

  print("*******")
  print("Test of the integrator with the problem")
  print("-(1/2)y''+(1/2)xy=(3/4)y, y(0)=0.2, y'(0)=1.0")
  print("for which y(1)=0.934250481751617")
  print("")

  def pot(x):
   return x/2.0

  print("3 point solver")
  solver=Shoot_3p(pot, 0, 1)
  exact_ris=0.934250481751617

  print("{:>5s} {:>15s} {:>15s}".format("size", "err", "err*N^order"))
  
  solver.size=10
  for i in range(6):
    ris = solver.solveinitialproblem(3./4., 0.2, 1.0)
    scalerr=(exact_ris-ris)*pow(solver.size,solver.approxorder)
    print("{:>5d} {:>15.10f} {:>15.10f}".format(solver.size, exact_ris-ris, scalerr))
    solver.size*=2
  print()

  print("Numerov solver")
  solver=Shoot_Numerov(pot, 0, 1)
  exact_ris=0.934250481751617

  print("{:>5s} {:>15s} {:>15s}".format("size", "err", "err*N^order"))
  
  solver.size=5
  for i in range(6):
    ris = solver.solveinitialproblem(3./4., 0.2, 1.0)
    scalerr=(exact_ris-ris)*pow(solver.size,4)
    print("{:>5d} {:>15.10f} {:>15.10f}".format(solver.size, exact_ris-ris, scalerr))
    solver.size*=2
  print()
  print("*******")


  print("TEST: harmonic oscillator on [-10:10] (hbar=m=omega=1)")
  print()

  def pot(x):
    return x*x/2.0
  xmin=-10
  xmax=10

  print("3 point solver")
  solver=Shoot_3p(pot, xmin, xmax)
  print('{:>5s} {:>15s}'.format("size", "fund."))
  for size in range(50, 251, 50):
    ris=solver.solve(size,0.48)
    print('{:>5d} {:>15.10f}'.format(size, ris))
  print()

  print("Solution with tollerance of 1.0e-8: ", end='')
  ris=solver.solve_up_to_toll(0.48, 1.0e-8, sizeseed=100)
  print('{:>11.10f}'.format(ris))
  print()

  print("Numerov solver")
  solver=Shoot_Numerov(pot, xmin, xmax)
  print('{:>5s} {:>15s}'.format("size", "fund."))
  for size in range(50, 251, 50):
    ris=solver.solve(size,0.48)
    print('{:>5d} {:>15.10f}'.format(size, ris))
  print()

  print("Solution with tollerance of 1.0e-8: ", end='')
  ris=solver.solve_up_to_toll(0.49, 1.0e-8, sizeseed=100)
  print('{:>11.10f}'.format(ris))
  print()
  print("**********************")

