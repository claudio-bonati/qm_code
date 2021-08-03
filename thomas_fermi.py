#!/usr/bin/env python3

import math
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import scipy.optimize as optimize
import sys

__all__ = ["ThomasFermi"]

#***************************
#library functions and classes

class ThomasFermi:
  """Class for the numerical solution of the Thomas-Fermi equation
     y''=y^(3/2)/sqrt(t)
     y(0)=1, y(r->\infty)=0
  """


  def __init__(self, T=20):
    """ Solution on the interval [0:T]
    """
    self.step=None
    self.y0prime=None
    self.T=T


  def solveinitialproblem(self, y0prime, step):
    """Using t=x^2, \psi(x)=y(t) and \phi(x)=(d\psi/dx)/x
       we have

       d\psi/dx=2 x dy/dt
       d^2\psi/dx^2=\phi + 4x \psi^{3/2} (using T-F)

       hence (now primes denote d/dx)

       \psi'(x)=x \phi(x)
       \phi'(x)=4 \psi^{3/2}(x) (obtained from the definition of \phi)

       with intial conditions 

       \psi(0)=1
       \phi(0)=2*y0prime

       With these changes the equations to be integrated have C^1 r.h.s.
       We then itegrate these equations with the simple Euler scheme.
 
       This function returns [end, y(self.T)] with

       end  : 1 if y0prime is too small (and psi crosses zero)
              2 if y0prime is too large (and psi starts to increase)
       y(self.T) : estimated value of y(self.T) 
    """ 
    
    psi_i=1
    phi_i=2*y0prime
    x=0

    psi_ip1=psi_i+step*x*phi_i
    phi_ip1=phi_i+step*4*np.power(psi_i,1.5)
    x+=step

    tL=None
    tR=None

    end=0
    while(end==0):
      psi_i=psi_ip1
      phi_i=phi_ip1

      psi_ip1=psi_i+step*x*phi_i
      phi_ip1=phi_i+step*4*np.power(psi_i,1.5)
      x+=step

      if(x*x<self.T and (x+step)*(x+step)>=self.T):
        tL=x*x
        yL=psi_ip1

      if((x-step)*(x-step)<self.T and x*x>=self.T):
        tR=x*x
        yR=psi_ip1

      if(psi_ip1<0):
        end=1
      if(psi_ip1>psi_i):
        end=2

    if(tL!=None and tR!=None):
      yT=yL+(yR-yL)/(tR-tL)*(self.T-tL)
    else:
      yT=None    

    return end, yT 


  def solve_with_step(self, step, acc):
    """Find the ``critical'' value of the derivative in zero by using bisection.

       step : integration step to be used 
       acc  : relative accuracy of y(self.T) to be reached

       This function returns [y0prime, y(self.T)]
    """

    y0_L=-2
    risL, yTL = self.solveinitialproblem(y0_L, step)
    if(risL != 1):
      print("ERROR: risL must be smalle than the true value")
      sys.exit(1)

    y0_R=-1
    risR, yTR = self.solveinitialproblem(y0_R, step)
    if(risR != 2):
      print("ERROR: risR must be larger than the true value")
      sys.exit(1)

    x=(y0_L+y0_R)/2.0
    ris, yT =self.solveinitialproblem(x, step)
 
    while(yT==None or yTL==None or yTR==None):
      if(ris==2):
         y0_R=x
         yTR=yT
      else:
         y0_L=x
         yTL=yT

      x=(y0_L+y0_R)/2.0
      ris, yT =self.solveinitialproblem(x, step)

    while(np.abs((yTR-yTL)/(yTR+yTL)*2)>acc):
      if(ris==2):
         y0_R=x
         yTR=yT
      else:
         y0_L=x
         yTL=yT

      x=(y0_L+y0_R)/2.0
      ris, yT =self.solveinitialproblem(x, step)

    return x, yT


  def solve(self, acc, initialstep=2.0e-2):
    """Find the ``critical'' value of the derivative in zero by using bisection.

       acc : relative accuracy of y(self.T) to be reached
       initialstep : initial step used in the integration

       Returns:
       y0prime : the estimated critical value of y0prime
       step    : the integration step used
    """
  
    step1=initialstep
    ris1, yT_1 = self.solve_with_step(step1, acc)

    step2=initialstep/2
    ris2, yT_2 = self.solve_with_step(step2, acc)

    while(np.abs((yT_2-yT_1)/(yT_1+yT_2)*2)>acc):
      step1=step2
      ris1=ris2
      yT_1=yT_2

      step2=step1/2
      ris2, yT_2=self.solve_with_step(step2, acc)

    self.y0prime=ris2
    self.step=step2


  def get_spline_interp(self):
    """A spline interpolation of the solution is returned
    """
    if(self.y0prime==None or self.step==None):
      print("ERROR: solve has to be called before this function")
      sys.exit(1)

    listt=[]
    listpsi=[]

    psi_i=1
    phi_i=2*self.y0prime
    x=0
    listt.append(x*x)
    listpsi.append(psi_i)

    psi_ip1=psi_i+self.step*x*phi_i
    phi_ip1=phi_i+self.step*4*np.power(psi_i,1.5)
    x+=self.step
    listt.append(x*x)
    listpsi.append(psi_ip1)

    while(np.power(x+self.step,2)<self.T):
      psi_i=psi_ip1
      phi_i=phi_ip1

      psi_ip1=psi_i+self.step*x*phi_i
      phi_ip1=phi_i+self.step*4*np.power(psi_i,1.5)
      x+=self.step
      listt.append(x*x)
      listpsi.append(psi_ip1)

    interp=interpolate.interp1d(listt, listpsi, 'quadratic')
    return interp
   

#***************************
# unit testing


if __name__=="__main__":
  
  print("**********************")
  print("UNIT TESTING")
  print()

  print('Solve the Thomas-Fermi equation up to x=60 (x=rescaled variable)')
  print('with an accuracy of 1/10^4')
  print('')


  test=ThomasFermi(T=60)
  test.solve(1.0e-4)
  interp=test.get_spline_interp()

  for x in np.arange(0, 50.0001, 0.01):
    print(' {:>5f} {:15.10f}'.format(x, interp(x)))

  b=np.power(3./4.*np.pi, 2./3.)/2.  ##\approx 0.885
  def density(r):  ## from Landau 3 eq. 70.9
    return 32./9./np.power(np.pi,3) * np.power(interp(r/b)/(r/b), 3./2.)
  def integrand(r):
    return 4*np.pi*r*r*density(r)

  print('Integral up to R=50 (atomic units) of the electron density')
  ris=integrate.quad(integrand, 0, 50)
  print(ris[0])
  print('should be 1 but the solution has an heavy tail...')
  print('')
 
  def func_to_vanish(r):
    ris=integrate.quad(integrand, 0, r)
    return ris[0]-0.5

  print('Radius (in atomic units) contaning 50% of the charge for Z=1')
  ris=optimize.newton(func_to_vanish, 1.4) 
  print(ris)
  print('According to Landau this should be approx 1.33')
  print('According to Galindo and Pasqual it should be approx 1.682')
  print('Note that Table 2 of par.70 of Landau 3 is nicely reproduced by our data')
 




  print("**********************")

