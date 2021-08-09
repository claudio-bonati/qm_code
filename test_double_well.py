#!/usr/bin/env python3

import numpy as np
import sch_solver_dense as dense
import sch_solver_shoot as shoot
from scipy import optimize
import sys

if __name__=="__main__":

  # Compute the first four levels of the double well 

  # single well of depth V04 and size \approx a
  # for |x|<<a we have well(x,V0,a)\approx -V0 + V0/(2 a^2)x^2
  def well(x, V0, a):
    return -V0*np.exp(-x*x/(2.0*a*a))


  # single well in b + single well in -b
  def double_well(x,V0,a,b):
    return well(x-b,V0,a) + well(x+b,V0,a)

  V0=15
  a=1
  b=0
  size=750 

  print('# Spectrum of single well of depth V0={:3>f} and width a={:2>f}'.format(2*V0, a))

  IRcut=optimize.newton(lambda x : well(x,2*V0,a)+0.05, x0=a, tol=1.0e-8)
  solver1=dense.FDM_Dense_5pD(lambda x : well(x, 2*V0, a), -IRcut, IRcut)
  aux=solver1.solve(size) 
  ris1=aux[0]
  ris2=aux[1]
  ris3=aux[2]
  ris4=aux[3]

  solver2=dense.FDM_Dense_5pD(lambda x : well(x, 2*V0, a), -2*IRcut, 2*IRcut)
  aux=solver2.solve(size) 
  ris5=aux[0]
  ris6=aux[1]
  ris7=aux[2]
  ris8=aux[3]

  print('# {:>15.10f} ({:>6.4e}) '.format(ris1, np.abs(ris5-ris1)), end='')
  print('{:>15.10f} ({:>6.4e}) '.format(ris2, np.abs(ris6-ris2)), end='')
  print('{:>15.10f} ({:>6.4e}) '.format(ris3, np.abs(ris7-ris3)), end='')
  print('{:>15.10f} ({:>6.4e}) '.format(ris4, np.abs(ris8-ris4)), end='')
  print('IRcut={:>4.2f} '.format(IRcut))
  print("# Harmonic hosc. approx. :", end='')
  print('{:>15.10f}'.format(-(2*V0)+np.sqrt((2*V0)/(a*a))*(0.5)), end='')
  print('{:>15.10f}'.format(-(2*V0)+np.sqrt((2*V0)/(a*a))*(1.5)), end='')
  print('{:>15.10f}'.format(-(2*V0)+np.sqrt((2*V0)/(a*a))*(2.5)), end='')
  print('{:>15.10f}'.format(-(2*V0)+np.sqrt((2*V0)/(a*a))*(3.5)))

  print()

  print('# Spectrum of single well of depth V0={:3>f} and width a={:2>f}'.format(V0, a))

  IRcut=optimize.newton(lambda x : well(x,V0,a)+0.05, x0=a, tol=1.0e-8)
  solver1=dense.FDM_Dense_5pD(lambda x : well(x, V0, a), -IRcut, IRcut)
  aux=solver1.solve(size) 
  ris1=aux[0]
  ris2=aux[1]
  ris3=aux[2]
  ris4=aux[3]

  solver2=dense.FDM_Dense_5pD(lambda x : well(x, V0, a), -2*IRcut, 2*IRcut)
  aux=solver2.solve(size) 
  ris5=aux[0]
  ris6=aux[1]
  ris7=aux[2]
  ris8=aux[3]

  print('# {:>15.10f} ({:>6.4e}) '.format(ris1, np.abs(ris5-ris1)), end='')
  print('{:>15.10f} ({:>6.4e}) '.format(ris2, np.abs(ris6-ris2)), end='')
  print('{:>15.10f} ({:>6.4e}) '.format(ris3, np.abs(ris7-ris3)), end='')
  print('{:>15.10f} ({:>6.4e}) '.format(ris4, np.abs(ris8-ris4)), end='')
  print('IRcut={:>4.2f} '.format(IRcut))
  print("# Harmonic hosc. approx. :", end='')
  print('{:>15.10f}'.format(-V0+np.sqrt((V0)/(a*a))*(0.5)), end='')
  print('{:>15.10f}'.format(-V0+np.sqrt((V0)/(a*a))*(1.5)), end='')
  print('{:>15.10f}'.format(-V0+np.sqrt((V0)/(a*a))*(2.5)), end='')
  print('{:>15.10f}'.format(-V0+np.sqrt((V0)/(a*a))*(3.5)))
  print()


  while b<3:
    IRcut=optimize.newton(lambda x : double_well(x,V0,a,b)+0.05, x0=b+a, tol=1.0e-8)

    solver1=dense.FDM_Dense_5pD(lambda x : double_well(x, V0, a,b), -IRcut, IRcut)
    aux=solver1.solve(size) 
    ris1=aux[0]
    ris2=aux[1]
    ris3=aux[2]
    ris4=aux[3]

    solver2=dense.FDM_Dense_5pD(lambda x : double_well(x, V0, a,b), -2*IRcut, 2*IRcut)
    aux=solver2.solve(size) 
    ris5=aux[0]
    ris6=aux[1]
    ris7=aux[2]
    ris8=aux[3]

    print('{:>.5f} {:>15.10f} {:>6.4e} '.format(b, ris1, np.abs(ris5-ris1)), end='')
    print('{:>15.10f} {:>6.4e} '.format(ris2, np.abs(ris6-ris2)), end='')
    print('{:>15.10f} {:>6.4e} '.format(ris3, np.abs(ris7-ris3)), end='')
    print('{:>15.10f} {:>6.4e} '.format(ris4, np.abs(ris8-ris4)), end='')
    print('{:>4.2f} '.format(IRcut))
    sys.stdout.flush() 

    b+=0.05
