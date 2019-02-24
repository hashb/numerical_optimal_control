## Implementation of the single shooting scheme of optimal control problems

## Create NLP
import casadi as C
import numpy as np
import pylab

N   = 30       # Control discretization
T   = 3.0      # End time
h   = T/N      # Discretization step
nx  = 1            
nu  = 1

Q = 1
R = 1

# Declare variables (use scalar graph)
u  = C.SX.sym('u',nu)    # control
x  = C.SX.sym('x',nx)    # states

# YOUR CODE HERE: System dynamics
# xdot = ...;

f = C.Function('f', [x,u],[xdot])

# RK4
U   = C.MX.sym('U')
X0  = C.MX.sym('X0',nx)
X   = X0

# YOUR CODE HERE: implement the RK4 integrator
RK4 = C.Function('RK4',[X0,U],[X])

# Formulate NLP (use matrix graph)
nv = N;
u = C.SX.sym('u',nv)

# Objective function
J=0;

# Get an expression for the cost and state at end
x_0 = 0.05;
X = x_0;
for i in range(N):
    # YOUR CODE HERE: build the cost of the NLP
    # 1 - get x_next using RK4
    # 2 - J = J + ...

# Terminal constraints: x_0(T)=x_1(T)=0
g = X

# Allocate an NLP solver
nlp = {'x': u, 'f': J, 'g': g}

# Create IPOPT solver object
options = {"ipopt": {"hessian_approximation":"limited-memory"}}
solver = C.nlpsol('solver', 'ipopt', nlp, options)

## Solve the NLP
arg = {}
# YOUR CODE HERE: upper and lower bounds on x and g(x)
# arg.x0  =  0.;        # solution guess
# arg.lbx =  ...;       # lower bound on x
# arg.ubx =  ...;       # upper bound on x
# arg.lbg =  ...;       # lower bound on g
# arg.ubg =  ...;       # upper bound on g


# Solve the problem
res = solver(arg)

f_opt       = res["f"].full()
u_opt       = res["x"].full()

# Compute state trajectory integrating the dynamics
x_opt       = np.zeros((nx,N+1))
x_opt[:,0]    = x_0

for i in range(1,N):
    out         = RK4([x_opt[:,i-1],u_opt[i-1]])
    x_opt[:,i]    = out[0].full()

## Plot results
pylab.figure()
pylab.subplot(2,1,1)
pylab.plot(x_opt.T)
pylab.xlabel('time - t')
pylab.ylabel('state - x')
pylab.grid(True)
pylab.subplot(2,1,2)
pylab.plot(u_opt,'r')
pylab.xlabel('time - t')
pylab.ylabel('input - u')
pylab.grid(True)
pylab.show()

