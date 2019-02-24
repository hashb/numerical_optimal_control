## Implementation of a Gauss-Neston SQP solver usign CasADi

# import CasADi
import casadi as C
import numpy as np

nv = 2
x = C.MX.sym('x',nv)

# YOUR CODE HERE: define the objective function (Task 5.2)
# f     = ...;
# F     = ...;
# Jf    = ...; 
x_test = np.array([1.1,3.2])
print F([x_test]) # 203.1300;
print Jf([x_test]) # [ -437.7000,  202.2000 ];

# YOUR CODE HERE: define the residuals (Task 5.2)
# r     = ...;
# R     = ...;
# Jr    = ...;
print R([x_test]) # [ 0.1000; 19.9000; 3.2000 ];
print Jr([x_test]) # [ 1,     0; -22,    10;  0,     1 ];

# Define equalities 
# YOUR CODE HERE: define the equality constraints 
# g     = ...;
# G     = ...;
# Jg    = ...;
print G([x_test]) # 5.9400;
print Jg([x_test]) # [ 1.0000,    4.4000];

# Define inequalities 
# YOUR CODE HERE: define the inequality constraints 
# h     = ...;
# H     = ...;
# Jh    = ...;
print H([x_test]) # -1.7900;
print Jh([x_test]) # [ 2.2000,   -1.0000 ];

# Define linearization point
xk = C.MX.sym('xk',nv)

# YOUR CODE HERE: define the linearized equalities 
# Jtemp     = Jg({xk});
# g_temp    = G({xk});
# g_l       = ...;

# YOUR CODE HERE: define the linearized inequalities 
# Jtemp     = Jh({xk});
# g_temp    = H({xk});
# h_l       = ...;

# YOUR CODE HERE: Gauss-Newton Hessian approximation (Task 5.3)
# j_out     = Jr({xk});
# jf_out    = Jf({xk});
# r_out     = R({xk});
# f_gn      = ...;

# Allocate QP solver
qp = {'x':x, 'f':f_gn,'p':xk}
solver = C.qpsol('solver', 'qpoases', qp)

#qp = {'x':x, 'f':f_gn,'g':g_l,'p':xk}
#solver = C.qpsol('solver', 'qpoases', qp)

#qp = {'x':x, 'f':f_gn,'g':C.vertcat([g_l,h_l]),'p':xk}
#solver = C.qpsol('solver', 'qpoases', qp)

# SQP solver
max_it = 100
xk = np.array([1,1]) # Initial guess

iters = [xk]
for i in range(1,max_it):    
    
    # YOUR CODE HERE: formulate the QP (Tasks 5.3, 5.4, 5.5)
    # arg     = {}
    # arg["lbg"] =  ...          
    # arg["ubg"] =  ...
    # arg["lbx"] =  ...          
    # arg["ubx"] =  ...
    # arg["p"]   =  ...

    # Solve with qpOASES
    sol = solver(arg)
    step = sol["x"]
    if np.linalg.norm(step) < 1.0e-16:
        break
    print step

    # Line-search 
    t = 1

    iters.append(iters[-1]+t*step)

[X,Y] = np.meshgrid(np.arange(-1.5,1.5,0.05), np.arange(-1.5,1.5,0.05))
Z = np.log(1 + 1.0/2*(X -1)**2 + 1.0/2*(10*(Y -X**2))**2 + 1.0/2*Y**2)

import pylab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pylab.figure()
pylab.subplot(1,2,1,projection='3d')
pylab.gca().plot_surface(X,Y,Z)
pylab.xlabel('x_1')
pylab.ylabel('x_2')

iters = np.array(C.horzcat(iters))


pylab.gca().plot(iters[0,:],iters[1,:],'k')
pylab.gca().plot(iters[0,:],iters[1,:],'ko')
y_g = np.linspace(-0.08,1.1,20)
x_g = -(1 - y_g)**2

pylab.gca().plot(x_g,y_g,'r')

x_h = np.linspace(-1.5,1.5,20)
y_h = 0.2 + x_h**2
pylab.gca().plot(x_h,y_h,'r--')

pylab.subplot(1,2,2)

pylab.contour(X,Y,Z)

pylab.plot(iters[0,:],iters[1,:],'k')
pylab.plot(iters[0,:],iters[1,:],'ko')
y_g = np.linspace(-0.08,1.1,20)
x_g = -(1 - y_g)**2

pylab.plot(x_g,y_g,'r')

x_h = np.linspace(-1.5,1.5,20)
y_h = 0.2 + x_h**2
pylab.plot(x_h,y_h,'r')

pylab.figure()
pylab.plot(iters.T)
pylab.xlabel('iterations')
pylab.ylabel('primal solution')

pylab.grid(True)
pylab.show()
