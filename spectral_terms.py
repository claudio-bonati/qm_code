#!/usr/bin/env python3

import itertools
import numpy as np
import sys

__all__ = ["terms_oneshell", "terms_twoshell"]

#***************************
#library functions

def place_ones(size, count):
  """Generate all different permutations of count ones and size-count zeros
  """

  for positions in itertools.combinations(range(size), count):
    #p=[0,...,0], len(p)=size
    p=[0]*size
  
    for i in positions:
      p[i] = 1

    yield p


def terms_oneshell(lvalue, numele):
  """Generate all the spectral terms of numele equivalent electrons in a
  shell with angular momentum ell
  """

  if(numele>2*(2*lvalue+1)):
    print("ERROR: too many electrons ({}) for a single level ({})".format(numele, lvalue))
    sys.exit(1)

  dict0={0:'s',1:'p',2:'d',3:'f',4:'g',5:'h',6:'i',7:'k',8:'l',9:'m',10:'n'}
  if(lvalue<=10):
    print("Spectral terms of {}^{} shell".format(dict0[lvalue], numele))
  else:
    print("Spectral terms of (L={})^{} shell".format(lvalue, numele))
  
  # use holes instead of electrons if more than half-filled 
  if(numele> 2*lvalue+1):
    numele=2*(2*lvalue+1)-numele

  #we use a list of 2(2*lvalue+1) numbers corresponding to
  #[{l_z=-lvalue,s_z=1/2},{l_z=-lvalue+1,s_z=1/2}.....{l_z=lvalue,s_z=1/2}, 
  #[{l_z=-lvalue,s_z=-1/2},{l_z=-lvalue+1,s_z=-1/2}.....{l_z=lvalue,s_z=-1/2}, 

  #create all differente permutations of numele ones and 2(2*lvalue+1)-numele zeros
  confs=list(place_ones(2*(2*lvalue+1),numele))

  #[-lvalue, ...,-1,0,1,....,lvalue]
  forprod=np.arange(-lvalue,lvalue+1,1,dtype=int)

  LSlist=[]
  # list of [L_z,2*S_z] values
  for element in confs:
    twoSz=np.sum(element[:2*lvalue+1])-np.sum(element[2*lvalue+1:])
    Lz=np.sum(np.multiply(np.add(element[:2*lvalue+1],element[2*lvalue+1:]),forprod))
    LSlist.append([Lz,twoSz])
  LSlist.sort(reverse=True) 
 
  terms=[]
  #list of [2S+1,L] values
  while len(LSlist) > 0:
    element=LSlist[0]

    L=element[0]
    twoSp1=element[1]+1
    terms.append(tuple([twoSp1,L]))

    #remove the identified multplet from the list
    for l in range(-L,L+1,1):
      for i in range(-element[1],element[1]+1,2):
        LSlist.remove([l,i])

  termscount=[[x,terms.count(x)] for x in set(terms)] 
  termscount.sort()   

  dict={0:'S',1:'P',2:'D',3:'F',4:'G',5:'H',6:'I',7:'K',8:'L',9:'M',10:'N'}

  for element in termscount:
    if(element[1]==1):
      if(element[0][1]<=10):
        print('{}{} '.format(element[0][0],dict[element[0][1]]), end='')  
      else:
        print('{}(L={}) '.format(element[0][0],element[0][1]), end='')  
    else:
      if(element[0][1]<=10):
        print('[{}]{}{} '.format(element[1],element[0][0],dict[element[0][1]]), end='')  
      else:
        print('[{}]{}(L={}) '.format(element[1],element[0][0],element[0][1]), end='')  

  print()


def terms_twoshell(lvalue1, numele1, lvalue2, numele2):
  """Generate all the spectral terms of two shells, with numele1 and numele2 equivalent electrons 
  with angular momenta ell1 and ell2 respectively
  """

  if(numele1>2*(2*lvalue1+1)):
    print("ERROR: too many electrons ({}) for a single level ({})".format(numele1, lvalue1))
    sys.exit(1)

  if(numele2>2*(2*lvalue2+1)):
    print("ERROR: too many electrons ({}) for a single level ({})".format(numele2, lvalue2))
    sys.exit(1)

  dict0={0:'s',1:'p',2:'d',3:'f',4:'g',5:'h',6:'i',7:'k',8:'l',9:'m',10:'n'}
  if(lvalue1<=10):
    if(lvalue2<=10):
      print("Spectral terms of {}^{} {}^{} shell".format(dict0[lvalue1],numele1,dict0[lvalue2], numele2))
    else:
      print("Spectral terms of {}^{} (L={})^{} shell".format(dict0[lvalue1],numele1,lvalue2, numele2))
  else:
    if(lvalue2<=10):
      print("Spectral terms of (L={})^{} {}^{} shell".format(lvalue1,numele1,dict0[lvalue2], numele2))
    else:
      print("Spectral terms of (L={})^{} (L={})^{} shell".format(lvalue1,numele1,lvalue2, numele2))
  
  # use holes instead of electrons if more than half-filled 
  if(numele1> 2*lvalue1+1):
    numele1=2*(2*lvalue1+1)-numele1
  if(numele2> 2*lvalue2+1):
    numele2=2*(2*lvalue2+1)-numele2

  #create all differente permutations of numele ones and 2(2*lvalue+1)-numele zeros
  confs1=list(place_ones(2*(2*lvalue1+1),numele1))
  confs2=list(place_ones(2*(2*lvalue2+1),numele2))

  confs=list(itertools.product(confs1,confs2))

  #[-lvalue, ...,-1,0,1,....,lvalue]
  forprod1=np.arange(-lvalue1,lvalue1+1,1,dtype=int)
  forprod2=np.arange(-lvalue2,lvalue2+1,1,dtype=int)

  LSlist=[]
  # list of [L_z,2*S_z] values
  for element in confs:
    element1=element[0]
    element2=element[1]

    twoSz =np.sum(element1[:2*lvalue1+1])-np.sum(element1[2*lvalue1+1:])  
    twoSz+=np.sum(element2[:2*lvalue2+1])-np.sum(element2[2*lvalue2+1:]) 
    Lz =np.sum(np.multiply(np.add(element1[:2*lvalue1+1],element1[2*lvalue1+1:]),forprod1)) 
    Lz+=np.sum(np.multiply(np.add(element2[:2*lvalue2+1],element2[2*lvalue2+1:]),forprod2))
    LSlist.append([Lz,twoSz])
  LSlist.sort(reverse=True) 
 
  terms=[]
  #list of [2S+1,L] values
  while len(LSlist) > 0:
    element=LSlist[0]

    L=element[0]
    twoSp1=element[1]+1
    terms.append(tuple([twoSp1,L]))

    #remove the identified multplet from the list
    for l in range(-L,L+1,1):
      for i in range(-element[1],element[1]+1,2):
        LSlist.remove([l,i])

  termscount=[[x,terms.count(x)] for x in set(terms)] 
  termscount.sort()   

  dict={0:'S',1:'P',2:'D',3:'F',4:'G',5:'H',6:'I',7:'K',8:'L',9:'M',10:'N'}

  for element in termscount:
    if(element[1]==1):
      if(element[0][1]<=10):
        print('{}{} '.format(element[0][0],dict[element[0][1]]), end='')  
      else:
        print('{}(L={}) '.format(element[0][0],element[0][1]), end='')  
    else:
      if(element[0][1]<=10):
        print('[{}]{}{} '.format(element[1],element[0][0],dict[element[0][1]]), end='')  
      else:
        print('[{}]{}(L={}) '.format(element[1],element[0][0],element[0][1]), end='')  

  print()


#***************************
# unit testing

if __name__=="__main__":
  
  print("**********************")
  print("UNIT TESTING")
  print()

  print('Notation of the result: [eventual multiplicity](2S+1)L')
  print()

  print('Single shell')
  ell=int(input("Insert the orbital momentum of the shell: "))
  ele=int(input("Insert the number of electrons in the shell: "))
  print()

  terms_oneshell(ell,ele)
  print()
  print()

  print('Two shell')
  ell1=int(input("Insert the orbital momentum of the shell 1: "))
  ele1=int(input("Insert the number of electrons in the shell 1: "))
  ell2=int(input("Insert the orbital momentum of the shell 2: "))
  ele2=int(input("Insert the number of electrons in the shell2 : "))

  print()

  terms_twoshell(ell1, ele1, ell2, ele2)
 
