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

DEBUG=False

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

       With these changes the equations to be integrated have C^1 r.h.s. and
       can be integrated with a standard Runge-Kutta 4th order integrator.
 
       This function returns [end, y(self.T)] with

       end  : 1 if y0prime is too small (and psi crosses zero)
              2 if y0prime is too large (and psi starts to increase)
       y(self.T) : estimated value of y(self.T) (None if the range of validity 
                   of the solution does not reach self.T) 
    """ 
   
    psi_i=1
    phi_i=2*y0prime
    x=0

    k_psi1=x*phi_i
    k_phi1=4*np.power(psi_i,1.5)

    k_psi2=(x+step/2)*(phi_i+step*k_phi1/2.0)
    if(psi_i+step*k_psi1/2.0<0):
      end=1
    k_phi2=4*np.power(psi_i+step*k_psi1/2.0,1.5)

    k_psi3=(x+step/2)*(phi_i+step*k_phi2/2.0)
    if(psi_i+step*k_psi2/2.0<0):
      end=1
    k_phi3=4*np.power(psi_i+step*k_psi2/2.0,1.5)

    k_psi4=(x+step)*(phi_i+step*k_phi3)
    if(psi_i+step*k_psi3<0):
      end=1
    k_phi4=4*np.power(psi_i+step*k_psi3,1.5)

    psi_ip1=psi_i+step*(k_psi1+2*k_psi2+2*k_psi3+k_psi4)/6.0
    phi_ip1=phi_i+step*(k_phi1+2*k_phi2+2*k_phi3+k_phi4)/6.0
    x+=step

    tL=None
    tR=None

    end=0
    while(end==0):
      psi_i=psi_ip1
      phi_i=phi_ip1

      k_psi1=x*phi_i
      k_phi1=4*np.power(psi_i,1.5)

      k_psi2=(x+step/2)*(phi_i+step*k_phi1/2.0)
      if(psi_i+step*k_psi1/2.0<0):
        end=1
        break
      k_phi2=4*np.power(psi_i+step*k_psi1/2.0,1.5)

      k_psi3=(x+step/2)*(phi_i+step*k_phi2/2.0)
      if(psi_i+step*k_psi2/2.0<0):
        end=1
        break
      k_phi3=4*np.power(psi_i+step*k_psi2/2.0,1.5)

      k_psi4=(x+step)*(phi_i+step*k_phi3)
      if(psi_i+step*k_psi3<0):
        end=1
        break
      k_phi4=4*np.power(psi_i+step*k_psi3,1.5)
  
      psi_ip1=psi_i+step*(k_psi1+2*k_psi2+2*k_psi3+k_psi4)/6.0
      phi_ip1=phi_i+step*(k_phi1+2*k_phi2+2*k_phi3+k_phi4)/6.0
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
      print("ERROR: y0_L must be smaller than the true value of the derivative in the origin")
      sys.exit(1)

    y0_R=-1
    risR, yTR = self.solveinitialproblem(y0_R, step)
    if(risR != 2):
      print("ERROR: y0_R must be larger than the true value of the derivative in the origin")
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
      if(DEBUG):
        print("debug1: ", y0_L, y0_R, y0_L-y0_R)

    index=0
    maxindex=50  # when the limit of double precision is reached the accuracy does not increase anymore. 
                 # This is the reason for the check on the number of iterations 
    while(np.abs((yTR-yTL)/(yTR+yTL)*2)>acc and index<maxindex):
      if(ris==2):
         y0_R=x
         yTR=yT
      else:
         y0_L=x
         yTL=yT

      x=(y0_L+y0_R)/2.0
      ris, yT =self.solveinitialproblem(x, step)

      index+=1

      if(DEBUG):
        print("debug1: ", y0_L, y0_R, y0_L-y0_R, np.abs((yTR-yTL)/(yTR+yTL)*2) )

    if(index==maxindex):
      print("Warning: maximum iteration reached")

    if(DEBUG):
      print("debug1: done")

    return x, yT


  def solve(self, acc, initialstep=2.0e-1):
    """Find the ``critical'' value of the derivative in zero by using bisection.

       acc : relative accuracy of y(self.T) to be reached
       initialstep : initial step used in the integration

       Returns:
       y0prime : the estimated critical value of y0prime
       step    : the integration step used
    """
  
    step1=initialstep
    ris1, yT_1 = self.solve_with_step(step1, acc)
    if(DEBUG):
      print("debug2: y0'=", ris1, ", yT=", yT_1, ", step=",step1) 
      print('')

    step2=initialstep/2
    ris2, yT_2 = self.solve_with_step(step2, acc)
    if(DEBUG):
      print("debug2: y0'=", ris2, ", yT=", yT_2, ", step=",step2) 
      print('')

    while(np.abs((yT_2-yT_1)/(yT_1+yT_2)*2)>acc):
      step1=step2
      ris1=ris2
      yT_1=yT_2

      step2=step1/2
      ris2, yT_2=self.solve_with_step(step2, acc)
      if(DEBUG):
        print("debug2: y0'=", ris2, ", yT=", yT_2, ", step=",step2) 
        print('')
    if(DEBUG):
      print("debug2: done")
      print('')

    self.y0prime=ris2
    self.step=step2


  def get_spline_interp(self, order=3):
    """A spline interpolation of the given order of the solution is returned
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

    k_psi1=x*phi_i
    k_phi1=4*np.power(psi_i,1.5)

    k_psi2=(x+self.step/2)*(phi_i+self.step*k_phi1/2.0)
    k_phi2=4*np.power(psi_i+self.step*k_psi1/2.0,1.5)

    k_psi3=(x+self.step/2)*(phi_i+self.step*k_phi2/2.0)
    k_phi3=4*np.power(psi_i+self.step*k_psi2/2.0,1.5)

    k_psi4=(x+self.step)*(phi_i+self.step*k_phi3)
    k_phi4=4*np.power(psi_i+self.step*k_psi3,1.5)

    psi_ip1=psi_i+self.step*(k_psi1+2*k_psi2+2*k_psi3+k_psi4)/6.0
    phi_ip1=phi_i+self.step*(k_phi1+2*k_phi2+2*k_phi3+k_phi4)/6.0
    x+=self.step
    listt.append(x*x)
    listpsi.append(psi_ip1)

    while(np.power(x+self.step,2)<self.T):
      psi_i=psi_ip1
      phi_i=phi_ip1

      k_psi1=x*phi_i
      k_phi1=4*np.power(psi_i,1.5)

      k_psi2=(x+self.step/2)*(phi_i+self.step*k_phi1/2.0)
      k_phi2=4*np.power(psi_i+self.step*k_psi1/2.0,1.5)

      k_psi3=(x+self.step/2)*(phi_i+self.step*k_phi2/2.0)
      k_phi3=4*np.power(psi_i+self.step*k_psi2/2.0,1.5)

      k_psi4=(x+self.step)*(phi_i+self.step*k_phi3)
      k_phi4=4*np.power(psi_i+self.step*k_psi3,1.5)
  
      psi_ip1=psi_i+self.step*(k_psi1+2*k_psi2+2*k_psi3+k_psi4)/6.0
      phi_ip1=phi_i+self.step*(k_phi1+2*k_phi2+2*k_phi3+k_phi4)/6.0
      x+=self.step
      listt.append(x*x)
      listpsi.append(psi_ip1)

    t=np.array(listt)
    psi=np.array(listpsi)

    interp=interpolate.InterpolatedUnivariateSpline(t, psi, k=order)
    return interp
   

#***************************
# unit testing


if __name__=="__main__":
  
  print("**********************")
  print("UNIT TESTING")
  print()

  Tmax=100
  acc=1.0e-5

  print('Solve the Thomas-Fermi equation up to x=',Tmax,'(x=rescaled variable)')
  print('with a relative accuracy of', acc)
  print('')

  test=ThomasFermi(T=Tmax)
  test.solve(acc)
  interp=test.get_spline_interp()

  #for x in np.arange(0, Tmax+0.0001, 0.01):
  #  print('{:5f} {:15.10f}'.format(x, float(interp(x))))

  b=np.power(3./4.*np.pi, 2./3.)/2.  ## approx 0.885
  def density(r):                    ## from Landau 3 eq. 70.9
     return 32./9./np.power(np.pi,3) * np.power(interp(r/b)/(r/b), 3./2.)
  def integrand(r):
    # 4 pi r^2 density(r)
    return 4*np.pi*np.sqrt(r)* 32./9./np.power(np.pi,3) * np.power(interp(r/b)/(1.0/b), 3./2.)

  #for r in np.arange(0, b*Tmax+0.0001, 0.001):
  #  aux=integrate.quad(integrand,0,r)
  #  print('{:5f} {:15.10f} {:15.10f} {:15.10f}'.format(r, float(integrand(r)),float(aux[0]), float(aux[1])))

  Rmax=80
  print('Integral up to R=', Rmax,'(atomic units) of the electron density')
  ris=integrate.quad(integrand, 0, Rmax)
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

