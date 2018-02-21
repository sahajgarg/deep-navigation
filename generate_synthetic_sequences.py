import numpy as np
import scipy.linalg
import cv2
import argparse
import os
##dynamics of this system are of the form
## x'' = -kx + ax'
##
## Reparametrize the system as p = [x, x']
## The dynamics are then
##
## p' = Ap = [0, I; -k, a]p
##
## with solution p(t) = e^(At)p(0)

## Takes in time t and initial state p0
def solve_disk(t, p0, runtime_config):
  A = np.zeros((4,4))
  A[0:2,2:4] = np.eye(2)
  A[2:4,0:2] = runtime_config.spring * np.eye(2)
  A[2:4,2:4] = runtime_config.drag * np.eye(2)
  At = t*A
  exp_At = scipy.linalg.expm(At)
  exp_At = np.asmatrix(exp_At)
  p0 = np.asmatrix(p0).T
  return np.dot(exp_At, p0)

def init_disk(r=None, c=None):
  p0 = np.random.rand(4)
  p0 -= 0.5
  p0 *= WINDOW_SIZE
  p0[2:4] = 0.0
  if r is None: r = np.random.rand(1)*25
  if c is None: c = np.random.rand(3)*255
  c = (int(c[0]), int(c[1]), int(c[2]))
  r = int(r)
  return p0, r, c

def draw_sol_onto_image(sol, col, img, path, rad):
  x = sol[0]
  y = sol[1]
  cv2.circle(img, (WINDOW_SIZE/2 + x, WINDOW_SIZE/2 + y), rad, col, -1)

def init_image():
  img = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3))
  img.fill(0)
  cv2.line(img, (WINDOW_SIZE/2, 0), (WINDOW_SIZE/2, WINDOW_SIZE-1), (0,0,0))
  cv2.line(img, (0, WINDOW_SIZE/2), (WINDOW_SIZE-1, WINDOW_SIZE/2), (0,0,0))
  return img

def draw_sols_onto_image(sols, cols, rads, path):
  img = init_image()
  for i in range(len(sols)): 
    draw_sol_onto_image(sols[i], cols[i], img, path, rads[i])
  cv2.imwrite(path, img)

def run_disk(runtime_config, red_disk=False):
  p0, r, c= init_disk()
  if red_disk: p0, r, c = init_disk(r=10.0,c=(0,0,255))
  sols = []
  for t in range(runtime_config.steps):
    sol = solve_disk(t, p0, runtime_config)
    noise = np.random.normal(runtime_config.mu, np.sqrt(runtime_config.sig2), (4,1))
    noise[2:4] = 0.0
    sol = sol + noise
    sols.append(sol)
    #draw_sols_onto_image([sol],[(0,0,255)], "./imgs/" + str(t) + ".png")
  return (sols, r, c)

def run_and_save_disks(runtime_config):
  n = runtime_config.n
  sol_sets = [run_disk(runtime_config) for i in range(n)]
  red_set = run_disk(runtime_config, red_disk=True)
  sol_sets.append(red_set)

  ##draw images
  cols = [elem[2] for elem in sol_sets]
  rads = [elem[1] for elem in sol_sets]
  time_sols = []
  for t in range(runtime_config.steps):
    sols_t = [elem[0][t] for elem in sol_sets]
    time_sols.append(sols_t)
  for t in range(runtime_config.steps): 
    draw_sols_onto_image(time_sols[t], cols, rads, "./imgs/" + str(t) + ".png")

  ##write out trajectory
  red_output = np.hstack(red_set[0])
  np.save("red.npy", red_output)

SPRING_CONSTANT = -5.0
DRAG_CONSTANT = -1.0
DIM = 2
WINDOW_SIZE = 512 ## 256 x 256 images 
MU = 0.0
SIG2 = 25.0
N = 10
STEPS = 100
class RConfig:
  def __init__(self, args):
    self.spring = SPRING_CONSTANT
    self.drag = DRAG_CONSTANT
    self.n = N
    self.mu = MU
    self.sig2 = SIG2
    self.steps = STEPS
    if 'spring-const' in args: self.spring = args['spring-const']
    if 'drag-const' in args: self.drag = args['drag-const']
    if 'mean-noise' in args: self.mu = args['mean-noise']
    if 'std-noise' in args: self.sig2 = args['std-noise']
    if 'num-disks' in args: self.n = args['num-disks']
    if 'time-steps' in args: self.steps = args['time-steps']

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--spring-const', '-k', type=float, help="Spring constant (default -5)")
  parser.add_argument('--drag-const', '-d', type=float, help="Drag constant (default -1)")
  parser.add_argument('--mean-noise', '-mu', type=float, help="Mean of Gaussian noise")
  parser.add_argument('--std-noise', '-sig2', type=float, help="var of Gaussian noise")
  parser.add_argument('--num-disks', '-n', type=int, help="Number of disks to render")
  parser.add_argument('--time-steps', '-t', type=int, help="Number of steps to sample")
  arg_dict = vars(parser.parse_args())
  runtime_config = RConfig(arg_dict)  

  if not os.path.exists("imgs"): 
    os.makedirs("imgs")
  else:
    for files in os.listdir("imgs"): os.remove("imgs/" + files)
  run_and_save_disks(runtime_config)

if __name__ == "__main__": main()
