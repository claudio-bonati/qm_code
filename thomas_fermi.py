#!/usr/bin/env python3

import math
import numpy as np
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


  def solveinitialproblem(self, y0prime, step, printvalues=False):
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
    if(printvalues):
      print('{:>15.10f} {:>15.10f} {:>15.10f}'.format(x*x, psi_i, phi_i/2))

    psi_ip1=psi_i+step*x*phi_i
    phi_ip1=phi_i+step*4*np.power(psi_i,1.5)
    x+=step
    if(printvalues):
      print('{:>15.10f} {:>15.10f} {:>15.10f}'.format(x*x, psi_ip1, phi_ip1/2))

    tL=None
    tR=None

    end=0
    while(end==0):
      psi_i=psi_ip1
      phi_i=phi_ip1

      psi_ip1=psi_i+step*x*phi_i
      phi_ip1=phi_i+step*4*np.power(psi_i,1.5)
      x+=step

      if(printvalues and (x-step)*(x-step)<=self.T):
        print('{:>15.10f} {:>15.10f} {:>15.10f}'.format(x*x, psi_ip1, phi_ip1/2))

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


  def print_values(self):
    """Print on screen the values of the solution of the T.F. equation and of its derivative
    """
    if(self.y0prime==None or self.step==None):
      print("ERROR: solve has to be called before this function")
      sys.exit(1)

    self.solveinitialproblem(self.y0prime, self.step, printvalues=True)
 

   

#***************************
# unit testing


if __name__=="__main__":
  
  print("**********************")
  print("UNIT TESTING")
  print()

  # Solve the T.F. equation up to T=20, with 
  # a relative accuracy of 1.0e-4 at T=20

  test=ThomasFermi(T=20)
  test.solve(1.0e-4)
  test.print_values()

  print("**********************")

