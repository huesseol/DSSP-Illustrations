import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
from pydsm.ft import dtft
import parser
from math import * #useful for parsing equation

filterG = "low pass"
cutoffFrequencyG = np.pi/2
impulseResponseG = '5 0 0 0 2'
transferFunctionG = '(z-1j)/(z**2 + z)'

def update_parameters(filter, cutoffFrequency, impulseResponse, transferFunction):
  global filterG, cutoffFrequencyG, impulseResponseG, transferFunctionG
  filterG, cutoffFrequencyG, impulseResponseG, transferFunctionG = filter, cutoffFrequency, impulseResponse, transferFunction

def widget():
    widgets.interact(update_parameters, 
      filter = widgets.Dropdown(
        options=['low pass', 'high pass', 'band pass', 'all pass', 'no pass', 'causal FIR', 'transfer function'],
        value='low pass',
        description='filter type'),
      cutoffFrequency = widgets.FloatSlider(
        value=np.pi/2,
        min=0,
        max=np.pi,
        description='cut off'),
      impulseResponse = widgets.Text(
        value='5 0 0 0 2',
        description='causal FIR'),
      transferFunction = widgets.Text(
        value='(z-1j)/(z**2 + z)',
        description='transfer function H(z) = ')
    )
    
#plot functional form
def display_filter_selection(continuous):
  plt.figure(figsize=(8, 0.4), dpi=80)
  if continuous:
      pass #so far not used, way of implementation left open
  else:
      if filterG == "low pass":
        plt.text(0.0, 0.0,'$H(\Omega) = I_{\{|\Omega| < cutoffFrequency\}}$', fontsize=22)
      elif filterG == "high pass":
        plt.text(0.0, 0.0,'$H(\Omega) = I_{\{|\Omega| \geq cutoffFrequency\}}$', fontsize=22)
      elif filterG == 'band pass':
        plt.text(0.0, 0.0,'$H(\Omega) = I_{\{|\Omega| \geq cutoffFrequency/2\}} \cdot I_{\{|\Omega| \leq (cutoffFrequency + \pi)/2\}}$', fontsize=22)
      elif filterG == 'all pass':
        plt.text(0.0, 0.0, '$H(\Omega) = 1$', fontsize=22)
      elif filterG == 'no pass':
        plt.text(0.0, 0.0, '$H(\Omega) = 0$', fontsize=22)
      elif filterG == 'causal FIR':
        plt.text(0.0, 0.0, 'taps = ' + impulseResponseG, fontsize=22)
      elif filterG == 'transfer function':
        plt.text(0.0, 0.0, 'H(z) = ' + transferFunctionG, fontsize=22)
        print("make sure that the transfer function is written in valid python code (e.g. ()^ denotes xor, ()**2 square). You may use function from the math library or numpy library (with prefix np.---)")
  plt.axis('off')
  plt.show()
  
#filter H
def H(Omega):
  val = None
  if filterG == "low pass":
    val = 1 if np.abs(Omega) < cutoffFrequencyG else 0
  elif filterG == "high pass":
    val = 0 if np.abs(Omega) < cutoffFrequencyG else 1
  elif filterG == "band pass":
    upper = (np.pi+cutoffFrequencyG)/2
    lower = cutoffFrequencyG/2
    val = 0 if np.abs(Omega) > upper or np.abs(Omega) < lower else 1
  elif filterG == "all pass":
    val = 1
  elif filterG == "no pass":
    val = 0
  elif filterG == "causal FIR":
      x = list(map(float, impulseResponseG.split()))
      val = dtft(x, fs = 2*np.pi, t0 = 0)(Omega) #sets f_s such that f=Omega as is done everywhere else, needs: from pydsm.ft import dtft
  elif filterG == "transfer function":
      z = np.exp(1j*Omega)
      code = parser.expr(transferFunctionG).compile()
      val = eval(code) #assumes code is some expression of z
  return val