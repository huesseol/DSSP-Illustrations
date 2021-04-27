import ipywidgets as widgets
import numpy as np

#size of window for signals
nG = np.arange(-20, 20)

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
        sinus_squared(a, sample)
    return sin

signalG = "unit step"
aG = 1

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
  return sig

def update_signal_parameters(signal, a):
  global signalG, aG
  signalG, aG = signal, a

def widget():
    widgets.interact(update_signal_parameters, 
      signal = widgets.Dropdown(
        options=['unit step', 'unit impulse', 'ramp', 'exponential', 'sinus', 'sinus squared'],
        value='unit step',
        description='signal type'),
      a = widgets.IntSlider(
        value=1,
        min=-10,
        max=10,
        step=1,
        description='parameter')
    )
