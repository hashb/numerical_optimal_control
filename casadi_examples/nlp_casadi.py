#! /usr/bin/env python

from casadi import *

# symbols and expressions
x = MX.sym('x')
y = MX.sym('y')
z = MX.sym('z')

f = x**2 + 100*z**2
g = z + (1-x)**2 - y

nlp = {}                      # NLP declaration
nlp['x'] = vertcat(x, y, z)   # decision vars
nlp['f'] = f                  # objective
nlp['g'] = g                  # constraints

# create solver instance
F = nlpsol('F', 'ipopt', nlp)

# solve the problem using an initial guess
F(x0=[2.5,3.0,0.75], ubg=0, lbg=0)
