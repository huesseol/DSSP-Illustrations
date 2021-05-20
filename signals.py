import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt

#size of window for signals
nG = np.arange(-20, 20)
nG3 = np.arange(-60, 60)
#several example signals

def unit_step_continuous(a, t):
    return 0 if t < a else 1

# n is a range type object describing the relevant domain, a describes the point where the step occurs
def unit_step(a, n):
    unit = []
    for sample in n:
        unit.append(unit_step_continuous(a, sample))
    return unit
    
def unit_impulse_continuous(a, t):
    epsilon = 0.001 #fidelity
    return 1/np.sqrt(epsilon * np.pi) * np.exp(- t**2 / epsilon)

# n is a range type object describing the relevant domain, a describes the point where the pulse occurs
def unit_impulse(a, n):
    delta = []
    for sample in n:
        if sample == a:
            delta.append(1)
        else:
            delta.append(0)
              
    return delta

def ramp_continuous(a, t):
    return 0 if t < a else (t-a)/4

# n is a range type object describing the relevant domain, a describes the point where the ramp begins
def ramp(a, n):
    ramp = []
    for sample in n:
        ramp.append(ramp_continuous(a, sample))
    return ramp

def exponential_continuous(a, t):
    return np.exp(a/20 * t)

# n is a range type object describing the relevant domain, a describes the growth rate
def exponential(a, n):
    expo = []
    for sample in n:
        expo.append(exponential_continuous(a, sample))
    return expo

def sinus_continuous(a, t):
    return np.sin(a/10 * t)

# n is a range type object describing the relevant domain, a describes the oscillation speed
def sinus(a, n):
    sin = []
    for sample in n:
        sin.append(sinus_continuous(a, sample))
    return sin
    
def sinus_squared_continuous(a, t):
    return sinus_continuous(a, t)**2
    
# n is a range type object describing the relevant domain, a describes the oscillation speed
def sinus_squared(a, n):
    sin = []
    for sample in n:
        sin.append(sinus_squared_continuous(a, sample))
    return sin

# n is a range type object describing the relevant domain
def causal_arbitrary(n):
    neg_len = len(n)//2
    
    sig = list(map(float, arbitrary_causal_signalG.split()))
    sig = [0]*neg_len + sig + [0]*(len(n) - neg_len - len(sig))
    return sig

signalG = "unit step"
aG = 1
arbitrary_causal_signalG = '1 2 1 -2 2'
#original signal in time domain
def time_domain_signal():
  sig = None
  if signalG == "unit step":
    sig = unit_step(aG, nG)
  elif signalG == "unit impulse":
    sig = unit_impulse(aG, nG)
  elif signalG == "ramp":
    sig = ramp(aG, nG)
  elif signalG == "exponential":
    sig = exponential(aG, nG)
  elif signalG == "sinus":
    sig = sinus(aG, nG)
  elif signalG == "sinus squared":
    sig = sinus_squared(aG, nG)
  elif signalG == "causal arbitrary":
    sig = causal_arbitrary(nG)
  return sig
  
#original signal in time domain (thrice the size)
def time_domain_signal_long():
  sig = None
  if signalG == "unit step":
    sig = unit_step(aG, nG3)
  elif signalG == "unit impulse":
    sig = unit_impulse(aG, nG3)
  elif signalG == "ramp":
    sig = ramp(aG, nG3)
  elif signalG == "exponential":
    sig = exponential(aG, nG3)
  elif signalG == "sinus":
    sig = sinus(aG, nG3)
  elif signalG == "sinus squared":
    sig = sinus_squared(aG, nG3)
  elif signalG == "causal arbitrary":
    sig = causal_arbitrary(nG3)
  return sig
  
#original signal in time domain, returns function handler
def time_domain_signal_continuous():
  sig = None
  if signalG == "unit step":
    sig = lambda t: unit_step_continuous(aG, t)
  elif signalG == "unit impulse":
    sig = lambda t: unit_impulse_continuous(aG, t)
  elif signalG == "ramp":
    sig = lambda t: ramp_continuous(aG, t)
  elif signalG == "exponential":
    sig = lambda t: exponential_continuous(aG, t)
  elif signalG == "sinus":
    sig = lambda t: sinus_continuous(aG, t)
  elif signalG == "sinus squared":
    sig = lambda t: sinus_squared_continuous(aG, t)
  elif signalG == "causal arbitrary":
    sig = lambda t: 0 #returns 0 since not continuous
  return sig

def update_signal_parameters(signal, a, arbitrary_causal_signal):
  global signalG, aG, arbitrary_causal_signalG
  signalG, aG, arbitrary_causal_signalG = signal, a, arbitrary_causal_signal

#plot functional form
def display_signal_selection(continuous):
  plt.figure(figsize=(8, 0.4), dpi=80)
  if continuous:
      if signalG == "unit step":
        plt.text(0.0, 0.0,'$u(t) = \sigma(t-parameter)$', fontsize=22)
      elif signalG == "unit impulse":
        plt.text(0.0, 0.0,'$u(t) = \delta(t-parameter)$', fontsize=22)
      elif signalG == "ramp":
        plt.text(0.0, 0.0,'$u(t) = (t-parameter)/4 \cdot \sigma(t-parameter)$', fontsize=22)
      elif signalG == "exponential":
        plt.text(0.0, 0.0,'$u(t) = \exp (t \cdot parameter/20)$', fontsize=22)
      elif signalG == "sinus":
        plt.text(0.0, 0.0,'$u(t) = \sin (t \cdot parameter/10)$', fontsize=22)
      elif signalG == "sinus squared":
        plt.text(0.0, 0.0,'$u(t) = \sin^2 (t \cdot parameter/10)$', fontsize=22)
      elif signalG == 'causal arbitrary':
          pass #only discrete time signal
  else:
      if signalG == "unit step":
        plt.text(0.0, 0.0,'$u[k] = \sigma[k-parameter]$', fontsize=22)
      elif signalG == "unit impulse":
        plt.text(0.0, 0.0,'$u[k] = \delta[k-parameter]$', fontsize=22)
      elif signalG == "ramp":
        plt.text(0.0, 0.0,'$u[k] = (k-parameter)/4 \cdot \sigma[k-parameter]$', fontsize=22)
      elif signalG == "exponential":
        plt.text(0.0, 0.0,'$u[k] = \exp (k \cdot parameter/20))$', fontsize=22)
      elif signalG == "sinus":
        plt.text(0.0, 0.0,'$u[k] = \sin (k \cdot parameter/10)$', fontsize=22)
      elif signalG == "sinus squared":
        plt.text(0.0, 0.0,'$u[k] = \sin^2 (k \cdot parameter/10)$', fontsize=22)
      elif signalG == 'causal arbitrary':
        plt.text(0.0, 0.0,'taps = ' + arbitrary_causal_signalG, fontsize=22)
  plt.axis('off')
  plt.show()

def widget():
    widgets.interact(update_signal_parameters, 
      signal = widgets.Dropdown(
        options=['unit step', 'unit impulse', 'ramp', 'exponential', 'sinus', 'sinus squared', 'causal arbitrary'],
        value='unit step',
        description='signal type'),
      a = widgets.IntSlider(
        value=1,
        min=-10,
        max=10,
        step=1,
        description='parameter'),
    arbitrary_causal_signal = widgets.Text(
        value='1 2 1 -2 2',
        description='causal signal')
    )
