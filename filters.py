import ipywidgets as widgets
import numpy as np

filterG = "low pass"
cutoffFrequencyG = np.pi/2

def update_parameters(filter, cutoffFrequency):
  global filterG, cutoffFrequencyG
  filterG, cutoffFrequencyG = filter, cutoffFrequency

def widget():
    widgets.interact(
      filter = widgets.Dropdown(
        options=['low pass', 'high pass', 'band pass', 'all pass', 'no pass'],
        value='low pass',
        description='filter type'),
      cutoffFrequency = widgets.FloatSlider(
        value=np.pi/2,
        min=0,
        max=np.pi,
        description='cut off')
    )
  
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
  return val