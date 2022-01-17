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

# For some mathematical giustification of the method used see e.g.
# https://www.pi.infn.it/~bonati/fisica3_2021/thomas_fermi.pdf

class ThomasFermi:
  """Class for the numerical solution of the Thomas-Fermi equation
     chi''(x)=chi^{3/2}(x)/sqrt(x)
     chi(0)=1, chi(r->\infty)=0
  """


  def __init__(self, rmax=20):
    """ Solution on the interval [0:rmax]
    """
    self.step=None
    self.chi0prime=None
    self.chi0prime_err=None
    self.rmax=rmax


  def solveinitialproblem(self, chi0prime, step):
    """Using t=x^2, y(t)=chi(x) and z(x)=2 d\chi(x)/dx
       we have

       y'(t)=t*z(t)
       z'(t)=4*y^{3/2}(t)

       with intial conditions 

       y(0)=1
       z(0)=2*chi0prime

       With these changes the equations to be integrated have C^1 r.h.s. and
       can be integrated with a standard Runge-Kutta 4th order integrator.
 
       Returns
       end            : 1 if chi0prime is too small (and y(t) crosses zero)
                        2 if chi0prime is too large (and y(t) starts to increase)
       chi(self.rmax) : estimated value of chi(self.rmax) (None if the range of 
                        validity of the solution does not reach self.rmax) 
    """ 
  
    y_i=1
    z_i=2*chi0prime
    t=0

    k_y1=t*z_i
    k_z1=4*np.power(y_i,1.5)

    k_y2=(t+step/2)*(z_i+step*k_z1/2.0)
    if(y_i+step*k_y1/2.0<0):
      end=1
    k_z2=4*np.power(y_i+step*k_y1/2.0,1.5)

    k_y3=(t+step/2)*(z_i+step*k_z2/2.0)
    if(y_i+step*k_y2/2.0<0):
      end=1
    k_z3=4*np.power(y_i+step*k_y2/2.0,1.5)

    k_y4=(t+step)*(z_i+step*k_z3)
    if(y_i+step*k_y3<0):
      end=1
    k_z4=4*np.power(y_i+step*k_y3,1.5)

    y_ip1=y_i+step*(k_y1+2*k_y2+2*k_y3+k_y4)/6.0
    z_ip1=z_i+step*(k_z1+2*k_z2+2*k_z3+k_z4)/6.0
    t+=step

    #xL will be the point closer to self.rmax on the left 
    #xR will be the point closer to self.rmax on the right
    xL=None
    xR=None

    end=0
    while(end==0):
      y_i=y_ip1
      z_i=z_ip1

      k_y1=t*z_i
      k_z1=4*np.power(y_i,1.5)

      k_y2=(t+step/2)*(z_i+step*k_z1/2.0)
      if(y_i+step*k_y1/2.0<0):
        end=1
        break
      k_z2=4*np.power(y_i+step*k_y1/2.0,1.5)

      k_y3=(t+step/2)*(z_i+step*k_z2/2.0)
      if(y_i+step*k_y2/2.0<0):
        end=1
        break
      k_z3=4*np.power(y_i+step*k_y2/2.0,1.5)

      k_y4=(t+step)*(z_i+step*k_z3)
      if(y_i+step*k_y3<0):
        end=1
        break
      k_z4=4*np.power(y_i+step*k_y3,1.5)
  
      y_ip1=y_i+step*(k_y1+2*k_y2+2*k_y3+k_y4)/6.0
      z_ip1=z_i+step*(k_z1+2*k_z2+2*k_z3+k_z4)/6.0
      t+=step

      if(t*t<self.rmax and (t+step)*(t+step)>=self.rmax):
        xL=t*t
        yL=y_ip1

      if((t-step)*(t-step)<self.rmax and t*t>=self.rmax):
        xR=t*t
        yR=y_ip1

      if(y_ip1<0):
        end=1
      if(y_ip1>y_i):
        end=2
 
    if(xL!=None and xR!=None):
      yrmax=yL+(yR-yL)/(xR-xL)*(self.rmax-xL)
    else:
      yrmax=None    

    return end, yrmax 


  def solve_with_step(self, step, acc):
    """Find the ``critical'' value of the derivative in zero by using bisection.

       step : integration step to be used 
       acc  : relative accuracy of chi(self.rmax) to be reached

       Returns:
       chi0prime      : the estimated critical value of chi0prime
       chi0prime_err  : the estimated error on the value of chi0prime
       chi(self.rmax) : the estimated value of the soluzion in self.rmax
    """
    if(DEBUG):
      print("  debug1:  inside solve_with_step ", step, acc)

    #here L stands for lower-bound, U for upper bound

    yp0_L=-2
    risL, yrmaxL = self.solveinitialproblem(yp0_L, step)
    if(risL != 1):
      print("ERROR: y0_L must be smaller than the true value of the derivative in the origin")
      sys.exit(1)

    yp0_U=-1
    risU, yrmaxU = self.solveinitialproblem(yp0_U, step)
    if(risU != 2):
      print("ERROR: y0_U must be larger than the true value of the derivative in the origin")
      sys.exit(1)

    yp0=(yp0_L+yp0_U)/2.0
    ris, yrmax =self.solveinitialproblem(yp0, step)
 
    while(yrmax==None or yrmaxL==None or yrmaxU==None):
      if(ris==2):
         yp0_U=yp0
         yrmaxU=yrmax
      else:
         yp0_L=yp0
         yrmaxL=yrmax

      yp0=(yp0_L+yp0_U)/2.0
      ris, yrmax =self.solveinitialproblem(yp0, step)
      if(DEBUG):
        print("  debug1a: ", yp0_L, yp0_U, yp0_U-yp0_L)

    index=0
    maxindex=50  # when the limit of double precision is reached the accuracy does not increase anymore. 
                 # This is the reason for the check on the number of iterations 
    while(np.abs(2*(yrmaxU-yrmaxL)/(yrmaxU+yrmaxL))>acc and index<maxindex):
      if(ris==2):
         yp0_U=yp0
         yrmaxU=yrmax
      else:
         yp0_L=yp0
         yrmaxL=yrmax

      yp0=(yp0_L+yp0_U)/2.0
      ris, yrmax =self.solveinitialproblem(yp0, step)

      index+=1

      if(DEBUG):
        print("  debug1b: ", yp0_L, yp0_U, yp0_U-yp0_L, np.abs(2*(yrmaxU-yrmaxL)/(yrmaxU+yrmaxL)) )

    yp0err=yrmaxU-yrmaxL

    if(index==maxindex):
      print("Warning: maximum iteration reached")

    if(DEBUG):
      print("  debug1:  done")
      print("")

    return yp0, yp0err, yrmax


  def solve(self, acc, initialstep=2.0e-1):
    """Find the ``critical'' value of the derivative in zero by using bisection.

       acc : relative accuracy of y(self.T) to be reached
       initialstep : initial step used in the integration

       Fix:
       chi0prime     : the estimated critical value of chi0prime
       chi0prime_err : the estimated error on the value of chi0prime
       step          : the integration step used
    """
  
    if(DEBUG):
      print("debug2:  inside solve", acc, initialstep) 

    step1=initialstep
    ris1, ris1err, yrmax1 = self.solve_with_step(step1, acc)
    if(DEBUG):
      print("debug2: y0'=", ris1, ", yrmax=", yrmax1, ", step=", step1) 
      print('')

    step2=initialstep/2
    ris2, ris2err, yrmax2 = self.solve_with_step(step2, acc)
    if(DEBUG):
      print("debug2: y0'=", ris2, ", yrmax=", yrmax2, ", step=", step2) 
      print('')

    while(np.abs((yrmax2-yrmax1)/(yrmax1+yrmax2)*2)>acc):
      step1=step2
      ris1=ris2
      yrmax1=yrmax2

      step2=step1/2
      ris2, ris2err, yrmax2=self.solve_with_step(step2, acc)
      if(DEBUG):
        print("debug2: y0'=", ris2, ", yrmax=", yrmax2, ", step=", step2) 
        print('')
    if(DEBUG):
      print("debug2: done")
      print('')

    # the error associated with chi0prime is the quadrature sum of the
    # error depending on the step and the error at fixed step
    self.chi0prime_err=np.sqrt((ris2-ris1)*(ris2-ris1)+ris2err*ris2err)
    self.chi0prime=ris2
    self.step=step2


  def get_spline_interp(self, order=3):
    """A spline interpolation of the given order of the solution is returned
    """
    if(self.chi0prime==None or self.step==None):
      print("ERROR: solve has to be called before this function")
      sys.exit(1)

    # remember that x=t*t and chi(x)=y(t)
    listx=[]
    listchi=[]

    y_i=1
    z_i=2*self.chi0prime
    t=0

    listx.append(t*t)
    listchi.append(y_i)

    k_y1=t*z_i
    k_z1=4*np.power(y_i,1.5)

    k_y2=(t+self.step/2)*(z_i+self.step*k_z1/2.0)
    k_z2=4*np.power(y_i+self.step*k_y1/2.0,1.5)

    k_y3=(t+self.step/2)*(z_i+self.step*k_z2/2.0)
    k_z3=4*np.power(y_i+self.step*k_y2/2.0,1.5)

    k_y4=(t+self.step)*(z_i+self.step*k_z3)
    k_z4=4*np.power(y_i+self.step*k_y3,1.5)

    y_ip1=y_i+self.step*(k_y1+2*k_y2+2*k_y3+k_y4)/6.0
    z_ip1=z_i+self.step*(k_z1+2*k_z2+2*k_z3+k_z4)/6.0
    t+=self.step

    listx.append(t*t)
    listchi.append(y_ip1)

    while(np.power(t+self.step,2)<self.rmax):
      y_i=y_ip1
      z_i=z_ip1

      k_y1=t*z_i
      k_z1=4*np.power(y_i,1.5)

      k_y2=(t+self.step/2)*(z_i+self.step*k_z1/2.0)
      k_z2=4*np.power(y_i+self.step*k_y1/2.0,1.5)

      k_y3=(t+self.step/2)*(z_i+self.step*k_z2/2.0)
      k_z3=4*np.power(y_i+self.step*k_y2/2.0,1.5)

      k_y4=(t+self.step)*(z_i+self.step*k_z3)
      k_z4=4*np.power(y_i+self.step*k_y3,1.5)
  
      y_ip1=y_i+self.step*(k_y1+2*k_y2+2*k_y3+k_y4)/6.0
      z_ip1=z_i+self.step*(k_z1+2*k_z2+2*k_z3+k_z4)/6.0
      t+=self.step

      listx.append(t*t)
      listchi.append(y_ip1)

    x=np.array(listx)
    chi=np.array(listchi)

    interp=interpolate.InterpolatedUnivariateSpline(x, chi, k=order)
    return interp
   

#***************************
# unit testing


if __name__=="__main__":
  
  print("**********************")
  print("UNIT TESTING")
  print()

  maxrange=100
  acc=1.0e-4

  print('Solve the Thomas-Fermi equation up to x={:.2f} (x=dimensionless variable)'.format(maxrange))
  print('with a relative accuracy {:.2e}'.format(acc))
  print('')

  # solve the Thomas-Fermi boundary value problem
  test=ThomasFermi(rmax=maxrange)
  test.solve(acc)

  print("Numrical solution details:")
  print(" stepsize : {:.4e}".format(test.step))
  print(" derivative in zero       : {:+.12f}".format(test.chi0prime))
  print(" derivative in zero error : {:+.12f}".format(test.chi0prime_err))
  print("")

  # a spline interpolation is used to simplify the numerical integration
  interp=test.get_spline_interp()

  Z=1
  b=np.power(3./4.*np.pi, 2./3.)/2.  # approx 0.885

  # from Landau 3 eq. 70.9
  def density(r):                  
     return Z*Z*32./9./np.power(np.pi,3) * np.power(interp(r*np.power(Z,1./3.)/b)/(r*np.power(Z,1./3.)/b), 3./2.)
  def integrand(r):
    return 4*np.pi*r*r*density(r)

  # Rmax<maxrange just to avoid spurious boundary effects
  Rmax=9*maxrange/10
  print('Integral up to R={:.2f} (atomic units) of the electron density (Z={:d})'.format(Rmax, Z))
  ris=integrate.quad(integrand, 0, Rmax)
  print(" {:.8f}".format(ris[0]))
  print('asymptotically should be {:d} but the solution has an heavy tail.'.format(Z))
  print('')
 
  def func_to_vanish(r):
    ris=integrate.quad(integrand, 0, r)
    return ris[0]-0.5*Z

  print('Radius (in atomic units) contaning 50% of the charge')
  ris=optimize.newton(func_to_vanish, 1.4) 
  print(" {:.8f}".format(ris))
  print("")

  print('According to Landau this should be approx 1.33/Z^(1./3.)={:.2f}'.format(1.33/np.power(Z,1./3.)))
  print("but this is likely a typo, since Table 2 of par.70 of Landau 3")
  print("is nicely reproduced. E.g.:")
  print(" {:5.2f} {:.6f}".format(0.06, float(interp(0.06))))
  print(" {:5.2f} {:.6f}".format(0.1, float(interp(0.1))))
  print(" {:5.2f} {:.6f}".format(0.5, float(interp(0.5))))
  print(" {:5.2f} {:.6f}".format(1,   float(interp(1))))
  print(" {:5.2f} {:.6f}".format(4,   float(interp(4))))
  print(" {:5.2f} {:.6f}".format(10,  float(interp(10))))
  print(" {:5.2f} {:.6f}".format(20,  float(interp(20))))
  print(" {:5.2f} {:.6f}".format(50,  float(interp(50))))





  print("**********************")

