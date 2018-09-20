#!/usr/bin/env python3

import fd_3p_d as fd
import numpy as np
from scipy import optimize
import sys

__all__ = ["shooting_solver_forw_3p", "shooting_solver_back_3p", "shooting_3p_d", "middle_shooting_3p_d"]

#***************************
#library functions

def shooting_solver_forw_3p(potential, xmin, xmax, N, E, y0, y0_prime):
  """Solver of the shooting equation (discretized using a 3 point discretization
  of the second derivative) with initial condition 
  
  psi(xmin)=y0
  psi'(xmin)=y0_prime

  potential = potential energy to be used
  the problem is defined on the interval [xmin, xmax]
  N = number of steps in which to divide [xmin, xmax]
  E = energy guess

  return the value of psi(x) and psi'(x) in x=xmax
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

  step=(xmax-xmin)/N

  # starting values at xmin
  psi_im1=y0
  psi_i=y0+y0_prime*step+(potential(xmin)-E)*y0*pow(step,2)/2

  for i in range(1, N, 1):
    psi_ip1=2*psi_i - psi_im1 - step*step*(E*psi_i-potential(xmin+i*step)*psi_i)

    if i==N-1:
     deriv=3*psi_ip1/2-2*psi_i+psi_im1/2

    psi_im1=psi_i 
    psi_i=psi_ip1

  return psi_i, deriv/step 


def shooting_solver_back_3p(potential, xmin, xmax, N, E, y0, y0_prime):
  """Solver of the shooting equation (discretized using a 3 point discretization
  of the second derivative) with initial condition 
  
  psi(xmax)=y0
  psi'(xmax)=y0_prime

  potential = potential energy to be used
  the problem is defined on the interval [xmin, xmax]
  N = number of steps in which to divide [xmin, xmax]
  E = energy guess

  return the value of psi(x) and psi'(x) in x=xmin
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

  step=(xmax-xmin)/N

  # starting values at xmin
  psi_im1=y0
  psi_i=y0-y0_prime*step+(potential(xmax)-E)*y0*pow(step,2)/2

  for i in range(1, N, 1):
    psi_ip1=2*psi_i - psi_im1 - step*step*(E*psi_i-potential(xmax-i*step)*psi_i)

    if i==N-1:
     deriv=-3*psi_ip1/2+2*psi_i-psi_im1/2

    psi_im1=psi_i 
    psi_i=psi_ip1

  return psi_i, deriv/step 



def shooting_3p_d(potential, xmin, xmax, initial_N, initial_E, tolerance, maxiter=500):
  """Shooting using 3 points discretization with Dirichlet boundary conditions.

  potential = potential energy to be used
  the problem is defined on the interval [xmin, xmax]
  initial_N = initial number of steps to be used
  initial_E = initial estimate for the eigenvalue
  tolerange = precision goal

  return the energy and the final value of steps used
  """

  locN=initial_N
  
  def f(x):
    ris, risprime = shooting_solver_forw_3p(potential, xmin, xmax, locN, x, 0.0, 1.0e-8)
    return ris

  #initial value
  ris0=optimize.newton(f, x0=initial_E, tol=tolerance)
  
  delta=1.0
  iteration=1

  while delta>tolerance and iteration < maxiter:
    locN*=2 
    ris1=optimize.newton(f, x0=ris0, tol=tolerance, maxiter=500)
    delta=abs(ris1-ris0)
    ris0=ris1
    iteration+=1

  if(iteration==maxiter):
    print("WARNING: maximum number of iterations reached")
    print("") 

  return ris0, locN


def middle_shooting_3p_d(potential, xmin, xmax, initial_N, initial_E, tolerance, maxiter=500):
  """Shooting in the middle using 3 points discretization with Dirichlet boundary conditions.

  potential = potential energy to be used
  the problem is defined on the interval [xmin, xmax]
  initial_N = initial number of steps to be used
  initial_E = initial estimate for the eigenvalue
  tolerange = precision goal

  return the energy and the final value of steps used
  """

  locN=initial_N
  
  def f(x):
    r1, rp1 = shooting_solver_forw_3p(potential, xmin, (xmax+xmin)/2.0, locN, x[0], 0.0, 1.0e-8)
    r2, rp2 = shooting_solver_back_3p(potential, (xmax+xmin)/2.0, xmax, locN, x[0], 0.0, x[1])
    return np.array([r1-r2, rp1-rp2])

  #initial value
  ris0=optimize.root(f, x0=np.array([initial_E, -1.0e-8]), tol=tolerance)
  
  delta=1.0
  iteration=1

  while delta>tolerance and iteration < maxiter:
    locN*=2 
    ris1=optimize.root(f, x0=ris0.x, tol=tolerance)
    delta=abs(ris1.x[0]-ris0.x[0])
    ris0=ris1
    iteration+=1

  if(iteration==maxiter):
    print("WARNING: maximum number of iterations reached")
    print("") 

  return ris0.x[0], locN



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

  N=200
  goal=1.0e-8

  risA=fd.fd_3p_d_solver(pot_h, xmin, xmax, N)

  print('Harmonic oscillator on [{:+.2f}, {:+.2f}]'.format(xmin, xmax))
  print('shooting using 3 points discretization and initial values from fd_3p_d_solver_extrap with N={:d}'.format(N))
  print('target precision {:g}'.format(goal))
  print('')
  print('{:>2s} {:>15s} {:>15s} {:>15s} {:>15s}'.format("n", "fd_3p_d", "shooting", "steps", "exact"))
  for i in range(0, 10, 2):
     ris, finalN=middle_shooting_3p_d(pot_h, xmin, xmax, N, risA[i], goal)
     print('{:>2d} {:>15.10f} {:>15.10f} {:>15d} {:>15.10f}'.format(i, risA[i], ris, finalN, 2*i+1))
  
  print("")
  print("")

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

  N=200
  goal=1.0e-8

  risA=fd.fd_3p_d_solver(pot_p, xmin, xmax, N)

  print('Paine problem (see Pryce "Numerical solution of Sturm-Liouville problems" App. A)')
  print('shooting using 3 points discretization and initial values from fd_3p_d_solver_extrap with N={:d}'.format(N))
  print('target precision {:g}'.format(goal))
  print('')
  print('{:>2s} {:>15s} {:>15s} {:>15s} {:>15s}'.format("n", "fd_3p_d", "shooting", "steps", "known"))
  for i in range(0, 10, 2):
     ris, finalN=middle_shooting_3p_d(pot_p, xmin, xmax, 10000, risA[i], goal)
     print('{:>2d} {:>15.10f} {:>15.10f} {:>15d} {:>15.10f}'.format(i, risA[i], ris, finalN, exact_p(i)))

  print("")
  print("")


  print("Test of the integrator with the problem -y''+xy=1.5y, y(0)=0.2, y'(0)=1.0")
  print("for which y(1)=0.934250481751617")
  print("")

  def pot_l(x):
    return x

  exact_ris=0.934250481751617

  print("{:>5s} {:>15s}".format("N", "err*N^2"))
  for N in range(20, 210, 20):
    ris, risprime = shooting_solver_forw_3p(pot_l, 0, 1, N, 1.5, 0.2, 1.0)
    print("{:>5d} {:>15.10f}".format(N, (exact_ris-ris)*pow(N,2)))

  print("**********************")
