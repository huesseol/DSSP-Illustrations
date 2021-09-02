#-------------------------------------------------------------------------------

#assumes the following global variables
#NG = len(yG)
#MG = [1, -1] #message is binary as standard
#yG = [1.1, -2, -0.7, 0.3]
#hG = [0.5, 1]

#Viterbi algorithm


#FILTER INVERSION ALGORITHM
#-------------------------------------------------------------------------------

import numpy as np

offset = 0
mem_r = {}


hG = None

def set_parameters(h):
    global hG
    hG = h

def g_right_sided_init():
  global offset, mem_r, hG
  offset = np.nonzero(hG)[0][0]
  print("offset m set to", offset)
  mem_r = {}


def g_right_sided_at(idx):
  m = offset
  n = -offset
  k = idx - n

  #computes a(l) using h(idx)
  def a(l):
    return h(l+m)
  #computes b(l) using g_right_sided_at(idx)
  def b(l):
    return g_right_sided_at(l+n)

  if k in mem_r:
    return mem_r[k]

  if k < 0:
    mem_r[k] = 0

  elif k == 0:
    mem_r[k] = 1/a(0)
  
  else: # k > 0
    sum = 0
    for i in range(0, k): #0, ..., k-1
      sum += a(k-i) * b(i)
    mem_r[k] = -sum/h(offset)

  return mem_r[k]
  
  
def h(k):
  if k in range(len(hG)):
    return hG[k]
  else:
    return 0