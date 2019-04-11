# Implementation of an interior point method 

import casadi as C
import numpy as np

# Problem definition
nv = 2
x = C.MX.sym('x',nv)

x_test = np.array([2,3])

# Inequality contstraints
ni = 1
# YOUR CODE HERE:
# Inequality contstraints
# H = Function(...);
print H([x_test]) # 9.0

ne = 1
# Equality contstraints
# G = Function(...);
print G([x_test]) # -8.0907;

# Objective
# Objective
# F = Function(...);
print F([x_test]) # 5;

# Create CasADi object that evaluates the Jacobian of g
# Jg = C.Function(...)
print Jg([x_test]) # [-0.4161,   -6.0000];

# Create CasADi object that evaluates the Jacobian of h
#Jh = C.Function(...)
print Jh([x_test]) # [4,   6];

# Create CasADi object that evaluates the Jacobian of f
#Jf = C.Function(...)
print Jf([x_test]) # [-4,   -2];

# Create CasADi object that evaluates the Hessian of the equalities
#Hg = C.Function(...)
print Hg([x_test]) #   [  -0.9093,  0        ; ...
                   #            0,        -2.0000 ];

# Create CasADi object that evaluates the Hessian of the inequalities
#Hh = C.Function(...)
print Hh([x_test]) #  [   2, 0; ...
                   #      0  2];

# Create CasADi object that evaluates the Hessian of the Lagrangian
#Hf = C.Function(...)
print Hf([x_test]) #  [   2, 0; ...
                   #      0  2];

# Interior point solver
max_it = 100
xk = np.array([[-2,-4]]).T
lk = 10*np.ones((ne,1))
vk = 10*np.ones((ni,1))
sk = 10*np.ones((ni,1))
iters = [C.vertcat([xk,lk,vk,sk])]

tau = 0
k_b = 1.0/3
th_1 = 1.0e-3
th_2 = 1.0e-16
for i in range(1,max_it):
    # Build KKT system
    [Hf_e]    = Hf([xk])
    [Hg_e]    = Hg([xk])
    [Hh_e]    = Hh([xk])
    Hl      = Hf_e + Hg_e*lk + Hh_e*vk
    
    [Jg_e]    = Jg([xk])
    [Jh_e]    = Jh([xk])
    [Jf_e]    = Jf([xk])
    
    [g_e]     = G([xk])
    [h_e]     = H([xk])
    
    # YOUR CODE HERE:
    # Buiild the KKT system
    # M = C.blockcat([[    ,    ... ,    ...  ,   ... ,    ...],
    #                   ... ,   ... ,    ... ,    ...],
    #                   ... ,    ... ,    ... ,    ...],
    #                   ... ,   ... ,    ...  ,   ...]])  
    
    # r = - C.vertcat([      ... + ... + ...   ,
    #                   ...               ,
    #                   ...               ,
    #                   ...               ])
    
    # Termination condition
    if np.linalg.norm(lhs) < th_1:
        if tau < th_2:
            print 'Solution found!'
            break
        else:
            tau = tau*k_b
        
    # YOUR CODE HERE:
    # Compute Newton step
    # sol = np.linalg.solve;
    
    # line-search
    max_ls = 100
    x_step  = sol[:nv]
    l_step  = sol[nv:nv+ne]
    v_step  = sol[nv+ne:nv+ne+ni]
    s_step  = sol[nv+ne+ni:]
    alpha = 1
    k_ls = 0.9
    min_step = 1.0e-8
    for j in range(max_ls):
        # YOUR CODE HERE: 
        # Compute trial step
        # l_t = ...;
        # s_t = ...;
        
        if (not np.any(v_t <= 0) and not np.any(s_t <= 0)):
            break

        # YOUR CODE HERE:
        # Decrease alpha
        # alpha = ...;
        
        if np.linalg.norm(alpha*np.vstack([ v_step,s_step])) < min_step:
            raise Exception('Line search failed! Could not find dual feasible step.')
            
    xk  = xk + alpha*x_step
    lk  = lk + alpha*l_step
    vk  = vk + alpha*v_step
    sk  = sk + alpha*s_step
    
    # Print some results
    print 'Iteration: ',i
    print 'tau = ', tau
    print 'norm(lhs) = ',np.linalg.norm(lhs)
    print 'step size = ', alpha
    iters.append(C.vertcat([xk,lk,vk,sk]))

iters = C.horzcat(iters)
import pylab
pylab.figure()
pylab.subplot(2,1,1)
pylab.plot(iters[:2,:].T)
pylab.grid(True)
pylab.xlabel('iterations')
pylab.ylabel('primal solution')
pylab.legend(('x_1','x_2'))
pylab.subplot(2,1,2)
pylab.plot(iters[2:4,:].T)
pylab.grid(True)
pylab.xlabel('iterations')
pylab.ylabel('dual solution')
pylab.legend(('mu','lambda'))

# Plot feasible set, and iterations
pylab.figure()
xs = np.linspace(-6,6,1000)
pylab.plot(xs,np.sqrt(np.sin(xs)),'r')
pylab.plot(xs,-np.sqrt(np.sin(xs)),'r')
pylab.plot(xs,np.sqrt(4-xs**2),'b')
pylab.plot(xs,-np.sqrt(4-xs**2),'b')

[X,Y] = np.meshgrid(np.linspace(-6,6), np.linspace(-6,6))
Z = (X- 4)**2 + (Y - 4)**2
pylab.contour(X,Y,Z)
pylab.plot(iters[0,:],iters[1,:],'ko')
pylab.title('Iterations in the primal space')
pylab.grid(True)

pylab.show()
