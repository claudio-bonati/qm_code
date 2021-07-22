#!/usr/bin/env python3

import numpy as np
import sch_solver_sparse as sparse
import sch_solver_shoot as shoot
from scipy import optimize
import sys

if __name__=="__main__":

 # Compute the first three levels of the anharmonic oscillator with \hbar=1, m=1 and omega=1
 # H=p^2/2+x^2/2+gx^4
 #  
 # Stability of the results is verified by doubling the discretization interval and these eigenvalues can 
 # be compared with high order perturbation theory, WKB, whatever... 

 def anharmonic_pot(x):
    return (1./2.)*x*x+g*pow(x,4)

 g=0.0
 while g<0.5:

   IRcut=optimize.newton(lambda x : anharmonic_pot(x)-50, x0=10, tol=1.0e-8)

   size=200

   solver=sparse.FDM_Sparse_5pD(anharmonic_pot, -IRcut, IRcut, 3)
   seed1, seed2, seed3=solver.solve(size) 
 
   solver=shoot.Shoot_Numerov(anharmonic_pot, -IRcut, IRcut)
   ris1=solver.solve_up_to_toll(seed1, 1.0e-8)
   ris2=solver.solve_up_to_toll(seed2, 1.0e-8)
   ris3=solver.solve_up_to_toll(seed3, 1.0e-8)

   solver=shoot.Shoot_Numerov(anharmonic_pot, -2*IRcut, 2*IRcut)
   ris4=solver.solve_up_to_toll(seed1, 1.0e-8)
   ris5=solver.solve_up_to_toll(seed2, 1.0e-8)
   ris6=solver.solve_up_to_toll(seed3, 1.0e-8)

   print('{:>.5f} {:>15.10f} {:>6.4e} '.format(g, ris1, np.abs(ris4-ris1)), end='')
   print('{:>15.10f} {:>6.4e} '.format(ris2, np.abs(ris5-ris2)), end='')
   print('{:>15.10f} {:>6.4e} '.format(ris3, np.abs(ris6-ris3)), end='')
   print('{:>4.2f} '.format(IRcut))
   sys.stdout.flush() 
   g+=0.02
