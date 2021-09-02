#LMS algorithm
#-------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

# input are numpy 1darrays: y, x, h_0 (initial weights); and a scalar: beta (learning rate)
# returns H (weight history) and E (Error history) (2darrays)
def LMS_adapt(y, x, h_0, beta):
  assert len(y) == len(x)
  assert len(h_0) >= 1
  assert len(y) >= len(h_0)

  H = np.array([h_0])
  E = np.array([])

  h = h_0

  for l in range(0, len(y) - len(h_0) + 1):
    yl = np.array([y[l + len(h_0)-1 - t] for t in range(0, len(h_0))]) #most recent values for y interval [l, l+len(h_0)-1] inverted
    el = x[l + len(h_0)-1] - np.dot(yl, h) #x[l + len(h_0)-1] - \hat{x}[l + len(h_0)-1]

    h = h + beta * el * yl
    
    H = np.append(H, [h], axis=0)
    E = np.append(E, [el], axis=0)

  return (H, E)
  
def plot_LMS(H, E):
    
  fig = plt.figure(figsize=(30, 8))

  ax1 = fig.add_subplot(1, 3, 1)
  ax1.grid()
  ax1.set_title('error of LMS during training')
  ax1.set_xlabel('$t$', fontsize=14)
  ax1.set_ylabel('$error[t]$', fontsize=14)
  ax1.axis(xmin=0, xmax=len(E))
  ax1.axis(ymin=-10, ymax=10)
  ax1.stem(E);

  ax2 = fig.add_subplot(1, 3, 2)
  ax2.grid()
  ax2.set_title('squared error of LMS during training')
  ax2.set_xlabel('$t$', fontsize=14)
  ax2.set_ylabel('$squared\ error[t]$', fontsize=14)
  ax2.axis(xmin=0, xmax=len(E))
  ax2.axis(ymin=-10, ymax=50)
  ax2.stem(E**2);

  ax3 = fig.add_subplot(1, 3, 3, projection='3d')
  ax3.grid()

  T, K = np.meshgrid(range(len(H[0])), range(len(H[:,1])))

  Xi = K.flatten()
  Yi = T.flatten()
  Zi = np.zeros(H.size)

  dx = 0.5 * np.ones(H.size)
  dy = 0.5 * np.ones(H.size)
  dz = H.flatten()

  ax3.set_title('filter coefficients evolution as time progresses')
  ax3.set_xlabel('time t')
  ax3.set_ylabel('index k')
  ax3.set_zlabel('$h_t[k]$')

  ax3.bar3d(Xi, Yi, Zi, dx, dy, dz)