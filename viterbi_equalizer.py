#since a lot of printing is done, this class allows to call methods of this file without printing. How to use:
#with HiddenPrints():
#    print("This will not be printed")
#
#print("This will be printed as before")

import os, sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

#-------------------------------------------------------------------------------

#assumes the following global variables
#NG = len(yG)
#MG = [1, -1] #message is binary as standard
#yG = [1.1, -2, -0.7, 0.3]
#hG = [0.5, 1]

#Viterbi algorithm
#-------------------------------------------------------------------------------

import numpy as np
import math
import copy

NG, MG, yG, hG = None, None, None, None

def set_parameters(N, M, y, h):
    global NG, MG, yG, hG
    NG, MG, yG, hG = N, M, y, h

def viterbi(B, S_i): #min-sum version
    """
    Returns the path of least path metric

    Parameters
    ----------
    B : 3D nested list, each outer element holds one section of branches (between 2 layers) encoded as a matrix (row-from, column-to). Inner elements encode metric of branch
    S_i: 1D list, initialize state metric of first layer

    Returns
    -------
    S : 2D nested list, lists state metrics layer by layer
    P : 2D nested list, lists minimizing prestates layer by layer
    """

    #check dimensionality of Trellis structure
    assert len(S_i) == len(B[0]), "S_i should fit the Trellis structure given by B in terms of dimensionality"
    for i in range(len(B)-1):
      assert len(B[i][0]) == len(B[i+1]), "the width of each layer of states should not depend if you look at it from the left or the right section of branches"

    #iterate through graph to compute state metrics
    #---------------------------------------------------------------------------

    #Initialize S and P
    S = [S_i] #list of states layer by layer with their metric
    P = [[None]*len(S_i)] #list of precursors for each state, same dimensions as S

    #iterate over sections to update states on the right side of it
    for section_index, section in enumerate(B):
      #layer of states to the right
      S_layer = []  #metrics
      P_layer = []  #prestates (0 indexed)
      for state_index in range(len(section[0])): #columns correspond to right layer
        state_metric = math.inf #unless better path is found
        state_prestate = None #unless better path is found

        #goes through prestates to find minimizer
        for prestate_index, prestate_metric in enumerate(S[section_index]):
          potential_metric = prestate_metric + section[prestate_index][state_index] #might be better than metric so far
          if potential_metric < state_metric:
            state_metric = potential_metric
            state_prestate = prestate_index
        S_layer += [state_metric]
        P_layer += [state_prestate]
      #add layer of state metrics and prestate indices
      S += [S_layer]
      P += [P_layer]
    
    return (S, P)
    
    
#helper functions to build Trellis
#-------------------------------------------------------------------------------

#x_last is an array of past entries X_{k-n}, ..., X_k
def z(x_last):
  assert len(x_last) == len(hG)
  return np.dot(x_last[::-1], hG) #convolution evaluated at newest element (x_last is inversely ordered to hG)

#w description is an array of past entries X_{k-n}, ..., X_k
def neg_log_likelihood_equalization(b_section, b_description):
  #actual neg_log_likelihood: \ln(\sqrt(2\pi) \sigma) + \frac{1}{2\sigma^2} (y_k - z(b_k))^2
  #purified from constants (y_k - z(b_k))^2 where 
  return (yG[b_section] - z(b_description))**2

def append_without_duplicates(l, e):
  l = copy.deepcopy(l)#do not change original list
  if e not in l:
    l.append(e)
  return l


#construct trellis states
#-------------------------------------------------------------------------------
def construct_trellis():
    #construct trellis states
    #-------------------------------------------------------------------------------
    
    S = [[ [0 for i in range(len(hG)-1)] ]]
    
    for layer in range(NG): #iterate over layers
      new_layer = []
      for state in S[layer]:
        for m in MG:
          x_rel = state + [m]
          new_layer = append_without_duplicates(new_layer, x_rel[1:])
      S.append(new_layer)
    
    print("state encodings")
    for layer in S:
      for state in layer:
        print("("+ ' '.join([str(e) for e in state]) + ")" , end =", ")
      print("")
      
    #construct trellis branch metrics
    #-------------------------------------------------------------------------------
    
    Br = []
    
    for layer in range(NG): #iterate over layers
      branches_matrix = []
      for left_state_idx in range(len(S[layer])):#iterates over left states
        branches_matrix.append([]) #for each row
        for right_state_idx in range(len(S[layer+1])): #iterates over next states
          x_rel = S[layer][left_state_idx] + [S[layer+1][right_state_idx][-1]]
          if x_rel[1:] == S[layer+1][right_state_idx]:
            branches_matrix[left_state_idx].append(neg_log_likelihood_equalization(layer, x_rel))
          else:
            branches_matrix[left_state_idx].append(math.inf)#branch shall not be taken, basically removes it
      Br.append(branches_matrix)
    
    #add final state such that prestates indicate best fit (prestate of final state only present if final state exists)
    Br.append([[0] for left_state_idx in S[-1]])
    
    print("branch metrics")
    for layer in Br[:-1]:
      print(layer)
      
    return S, Br
    
    
#invoke Viterbi algorithm
#-------------------------------------------------------------------------------
    
def invoke_viterbi(S, Br):
    #initialize state metric to 0 at start
    St_i = [0]
    #run viterbi algorithm (Sm are state metrics, Pn are prenodes)
    Sm, Pn = viterbi(Br, St_i)
    #reconstruct ideal path
    x = []
    state_idx = 0
    
    for idx in range(1, NG + 1):
      state_idx = Pn[-idx][state_idx]
      x.append(S[-idx][state_idx][-1])
    x = x[::-1]
    
    
    print("\nstate metrics evaluated")
    for layer in Sm[:-1]:
      print(layer)
    
    print("\nstate prenodes")
    for layer in Pn[1:]:
      print(layer)
    
    print("\nthe most likely estimation for x is")
    print(x)
        
    return x