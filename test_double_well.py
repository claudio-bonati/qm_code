#!/usr/bin/env python3

import numpy as np
import sch_solver_dense as dense
import sch_solver_shoot as shoot
from scipy import optimize
import sys

if __name__=="__main__":

  # Compute the first four levels of the double well 

  # single well of depth V0 and size \approx a
  # for |x|<<0 we have well(x,V0, a)\approx -V0/4 + V0/(16 a^2)x^2
  def well(x, V0, a):
    return -V0/(1.0+np.exp(x/a))/(1.0+np.exp(-x/a))  

  # single well in b + single well in -b
  def double_well(x,V0,a,b):
    return well(x-b,V0,a) + well(x+b,V0,a)

  V0=20
  a=1
 
  b=0
  while b<3.5:
    IRcut=optimize.newton(lambda x : double_well(x,V0,a,b)+0.1, x0=b+a, tol=1.0e-8)

    size=200

    solver=dense.FDM_Dense_5pD(lambda x : double_well(x, V0, a,b), -IRcut, IRcut)
    aux=solver.solve(size) 
    seed1=aux[0]
    seed2=aux[1]
    seed3=aux[2]
    seed4=aux[3]
 
    solver=shoot.Shoot_Numerov(lambda x : double_well(x,V0,a,b), -IRcut, IRcut)
    ris1=solver.solve_up_to_toll(seed1, 1.0e-8)
    ris2=solver.solve_up_to_toll(seed2, 1.0e-8)
    ris3=solver.solve_up_to_toll(seed3, 1.0e-8)
    ris4=solver.solve_up_to_toll(seed4, 1.0e-8)

    solver=shoot.Shoot_Numerov(lambda x : double_well(x,V0,a,b), -2*IRcut, 2*IRcut)
    ris5=solver.solve_up_to_toll(seed1, 1.0e-8)
    ris6=solver.solve_up_to_toll(seed2, 1.0e-8)
    ris7=solver.solve_up_to_toll(seed3, 1.0e-8)
    ris8=solver.solve_up_to_toll(seed4, 1.0e-8)

    print('{:>.5f} {:>15.10f} {:>6.4e} '.format(b, ris1, np.abs(ris5-ris1)), end='')
    print('{:>15.10f} {:>6.4e} '.format(ris2, np.abs(ris6-ris2)), end='')
    print('{:>15.10f} {:>6.4e} '.format(ris3, np.abs(ris7-ris3)), end='')
    print('{:>15.10f} {:>6.4e} '.format(ris4, np.abs(ris8-ris4)), end='')
    print('{:>4.2f} '.format(IRcut))
    sys.stdout.flush() 

    b+=0.02
