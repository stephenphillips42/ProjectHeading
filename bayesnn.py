import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

def activation(x):
  """Activation function we are using."""
  # return np.tanh(x)
  return np.maximum(0,x)

def get_network(weights):
  """Create MLP."""
  def fn(x):
    layer = x
    for i in range(len(weights["w"])):
      lin = np.dot(layer, weights["w"][i]) + weights["b"][i]
      layer = activation(lin)
    layer = np.dot(layer, weights["w_final"]) + weights["b_final"]
    return layer
    
  return fn

def generate_weights(sizes):
  """Return random network with specified by sizes.

  Args:
    sizes: list consisting of [input_size, hidden_size_1, ..., hidden_size_n, output_size]
  
  Returns:
    Dictionary with keys w, b, w_final, b_final specifying weights and biases"""
  weights = {}
  weights["w"] = []
  weights["b"] = []
  for i in range(len(sizes)-2):
    weights["w"].append(np.random.randn(sizes[i], sizes[i+1]))
    weights["b"].append(np.random.randn(sizes[i+1]))
  weights["w_final"] = np.random.randn(sizes[-2], sizes[-1])/np.sqrt(sizes[-1])
  weights["b_final"] = np.random.randn(sizes[-1])
  return weights
 
def my_fn(x):
  """Function we are estimating."""
  return 0.4*(0.5*(np.exp(x*4) - np.exp(-x*4)) - 8*x + 0.3*x**2 - 2*x**3 + 0.8)

def dmy_fn(x):
  """Function we are estimating."""
  return 0.4*(2.0*(np.exp(x*4) + np.exp(-x*4)) - 8 + 0.6*x - 6*x**2)
 
def norm_pdf(x, sigma):
  """PDF of multivariate normal distribution with covariance sigma."""
  return np.exp(-np.dot(x.T, np.linalg.solve(sigma,x))/2.0) / \
            np.sqrt(np.linalg.det(2*np.pi*sigma))

def log_norm_pdf(x, sigma):
  """log-PDF of multivariate normal distribution with covariance sigma."""
  _, det_sigma = np.linalg.slogdet(sigma)
  return -0.5*np.dot(x.T, np.linalg.solve(sigma,x))[0,0] - np.log(2*np.pi) - det_sigma

def log_pdf_weights(weights):
  """log-PDF of weights."""
  # Hidden units
  logpdf_w = []
  logpdf_b = []
  for i, w in enumerate(weights["w"]):
    logpdf_w.append(-0.5*np.dot(w.T, w)[0,0] - np.log(2*np.pi))
  for i, b in enumerate(weights["b"]):
    logpdf_b.append(-0.5*np.dot(b, b) - np.log(2*np.pi))
  # Final weights
  invs2 = (1./len(weights["w_final"]))
  wf = weights["w_final"]
  logpdf_wf = -0.5*np.dot(wf.T, invs2*wf)[0,0] - np.log(2*np.pi) + np.log(invs2)
  logpdf_bf = -0.5*np.dot(weights["b_final"].T, weights["b_final"]) - np.log(2*np.pi)
  return sum(logpdf_w) + sum(logpdf_b) + logpdf_wf + logpdf_bf

def log_add(x, y):
  """Adds numbers x and y in log space."""
  maximum = np.maximum(x,y)
  minimum = np.minimum(x,y)
  if(np.abs(maximum - minimum) > 30):
    # the difference is too small, return the just the maximum
    return maximum
  return maximum + np.log1p(np.exp(minimum - maximum))

def log_cum_sum(A, output_sum=False):
  """Cumulatively adds array of numbers x in log space."""
  C = [A[0]]
  for a in A[1:]:
    C.append(log_add(C[-1], a))
  C_norm = np.array(C) - C[-1]
  if output_sum:
    return C_norm, C[-1]
  else:
    return C_norm

def sample_from_log_prob(A, n):
  """Samples from array of (non-normalized) log probabilities x."""
  A_cum = log_cum_sum(A)
  C_pos = [ -c for c in reversed(A_cum)]
  sel = np.log(np.random.random(n))
  pos = [len(A) - np.searchsorted(C_pos,-r) for r in sel]
  return pos

def print_weight_info(weights):
  """Print informations about weights."""
  print("Length: {}\nw:\n{}\nb:\n{}".format(
        len(weights["w"]),
        [ w.shape for w in weights["w"] ],
        [ b.shape for b in weights["b"] ]))

# Constants
sizes = [1,16,1]
N_pts = 8
N_rand = 10
mysigma_ = 0.1
mysigma = mysigma_*np.eye(N_pts)
N_samples = 2**8*1000

# Generate true network
# w_true = generate_weights(sizes)
# y_fn = get_network(w_true)
y_fn = my_fn

# Generate points
areas = (np.random.uniform( 0.2, 0.8, (N_pts/2,)),
         np.random.uniform(-0.8,-0.2, (N_pts/2,)))
areas2 = (np.random.uniform( 0.2, 0.8, ((N_pts-2)/2,)),
         np.random.uniform(-0.8,-0.2, ((N_pts-2)/2,)),
         np.random.uniform( 0.8, 1.0, (1,) ),
         np.random.uniform(-1.0,-0.8, (1,) ))
x = np.concatenate(areas2).reshape(-1,1)
y = y_fn(x)

def main():
  # Generate random networks
  t_ = np.linspace(-1, 1, 400)
  t = t_.reshape(-1, 1)
  rand_networks = []
  rand_weights = []
  for i in range(N_rand):
    w = generate_weights(sizes)
    net = get_network(w)
    output = net(t)
    # Plot
    plt.plot(t,output)
    # Save for later
    rand_weights.append(w)
    rand_networks.append(net)
  plt.plot(t,y_fn(t))
  plt.scatter(x,y)
  plt.title("Random weights")
  plt.show()

  # Generate conditioned networks
  cond_networks = []
  cond_weights = []
  loglikelihood = []
  for i in tqdm(range(N_samples)):
    w = generate_weights(sizes)
    net = get_network(w)
    output = net(t)
    # Save for later
    cond_weights.append(w)
    cond_networks.append(net)
    loglikelihood.append(log_norm_pdf(net(x) - y, mysigma))
  # lcs, lcs_total = log_cum_sum(loglikelihood, True)
  # loglikelihood = np.array(loglikelihood) - lcs_total
  # print("N_samples")
  # plt.plot(sorted(loglikelihood))
  # plt.plot(sorted(np.log(np.random.random(N_samples))))
  # plt.show()
  # print("log cum sum")
  # plt.plot(lcs)
  # plt.show()
  # plt.plot(lcs[50:])
  # plt.show()
  # for i in range(15):
  #   h = (1.0*np.sum(loglikelihood < -i))/N_samples
  #   print("Greater than exp(-{}): {}".format(i,h))
  samples = sample_from_log_prob(loglikelihood, 10)
  for i, s in enumerate(samples):
    net = cond_networks[s]
    output = net(t) + np.random.randn()*0.1
    # Plot
    plt.plot(t,output)
  plt.plot(t,y_fn(t),'k')
  plt.scatter(x,y,c='k',zorder=10)
  plt.title("Conditioned weights")
  plt.show()

# Manually set up weights
step = 2.0/sizes[1]
weights = {}
weights["w"] = [np.array([ 1.0 ] * (sizes[1]/2) + [ -1.0 ] * (sizes[1]/2)).reshape(1,sizes[1])]
weights["b"] = [np.array([ -i*step for i in range(sizes[1]/2) ] * 2).reshape(sizes[1])]
dslope = lambda j: (my_fn(step*(j + 1)) - 2*my_fn(step*(j)) + my_fn(step*(j - 1)))
pos_slopes = [ (my_fn(step*(1.)) - my_fn(step*(0.)))/step ] + \
             [ dslope(i)/step for i in range(1,sizes[1]/2)]
neg_slopes = [ (my_fn(step*(-1.)) - my_fn(step*(0.)))/step ] + \
             [ dslope(-i)/step for i in range(1,sizes[1]/2)]
weights["w_final"] = np.array(pos_slopes + neg_slopes).reshape(sizes[1],1)
weights["b_final"] = my_fn(0)
net = get_network(weights)

nsteps = 8
t_ = np.linspace(-(nsteps - 1)*step, (nsteps - 1)*step, 400)
t_scatter = np.array([ i*step for i in range(-nsteps+1, nsteps)])
t = t_.reshape(-1, 1)
# plt.plot(t, my_fn(t))
# plt.plot(t, dmy_fn(t))
# plt.show()
plt.plot(t, net(t))
plt.plot(t, my_fn(t))
plt.scatter(t_scatter, my_fn(t_scatter))
plt.scatter(t_scatter, my_fn(t_scatter))
plt.show()
print((t_scatter, step))
print((my_fn(t_scatter), my_fn(step)))
print("Custom weights")
lgpdf = log_pdf_weights(weights)
print("Random weights")
randlgpdf = []
for i in range(60000):
  randlgpdf.append(log_pdf_weights(generate_weights(sizes)))
nn, bins, _ = plt.hist(randlgpdf, bins=200)
plt.plot([lgpdf, lgpdf], [0, np.max(nn)])
plt.show()

main()

