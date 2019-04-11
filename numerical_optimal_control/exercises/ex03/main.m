%           Exercise 3 - Explicit and Implict integrators.
% 
% In this exercise we are going to implement different integration scheme
% and analyze their properties. In particular, we will focus on stability
% and accuracy of explicit and implicit schemes with both numerical and
% theoretical taksks.

clear variables
close all
clc

% Define initial value problem
N   = 100;
Tf  = 10;
h   = Tf/N;
x0  = [1, 2]';
mu  = 1;
nx  = 2;

%% Reference solution: MATLAB stiffly accurate integrator
[T, solRef] = ode15s(@(t,x) van_der_pol(t,x,mu),(0:N-1)*h,x0);

% Plot trajectories
figure()
subplot(2,1,1)
title('Van der Pol (stiff) oscillator')
plot(T,solRef,'-')
hold all
legend('x_{1ref}','x_{2ref}')
grid on 
xlabel('time')
ylabel('ODE solution')
xlim([0,Tf])
ylim([-5,5])

%% Task 3.2 - Explicit Euler
solExpEu      = zeros(nx,N);
solExpEu(:,1) = x0;

for i = 2:N
    % YOUR CODE HERE: implement the explicit Euler integration scheme
end
plot(T,solExpEu,'--')
legend('x_{1r}','x_{2r}','x_{1ee}','x_{2ee}')

% Plot accuracy
subplot(2,1,2)
n_f = sum(solRef').^2;
n_f(abs(n_f) < 1) = 1;
semilogy(T,sum((solExpEu - solRef').^2)./n_f)
hold all
grid on 
xlabel('time')
ylabel('Accuracy')
legend('ee')
xlim([0,Tf])
ylim([1.0e-8,20])
ax = gca;
ax.YTick = [1.0e-7,1.0e-6,1.0e-5,1.0e-4,1.0e-3,1.0e-2,1.0e-1,1.0e-0];

%% Task 3.3 - Explicit Runge-Kutta 4
solRK4 = zeros(nx,N);
solRK4(:,1) = x0;

for i = 2:N
    solRK4(:,i) = RK4(solRK4(:,i-1),mu,@(x,p) van_der_pol(1,x,p),h);
end

% Plot solution
subplot(2,1,1)
plot(T,solRK4,':','LineWidth',1)
legend('x_{1r}','x_{2r}','x_{1ee}','x_{2ee}','x_{1rk}','x_{2rk}')

% Plot accuracy
subplot(2,1,2)
semilogy(T,sum((solRK4 - solRef').^2)./n_f)
legend('ee','rk')

%% Task 3.4 - Implicit Euler
import casadi.*  

s   = SX.sym('s',2);    % x_next
s0  = SX.sym('s0',2);   % x0
hp  = SX.sym('hp');     % parametric h
mup = SX.sym('mup');    % parametric mu

% YOUR CODE HERE: Define integrator's ( implicit ) equation
% f_expr = [        ...  ; ...
%                   .... ];
 
% Create a CasADi object that generates the Jacobian of f_expr
f = Function('f',{s0,s,hp,mup},{f_expr});
J = Function('J', {s0,s,hp,mup},{jacobian(f_expr,s)});

solIEU = zeros(nx,N);
solIEU(:,1) = x0;

for i = 2:N
    solIEU(:,i) = IEU(solIEU(:,i-1),mu,h,f,J);
end

% Plot solution
subplot(2,1,1)
plot(T,solIEU,'o')
legend('x_{1r}','x_{2r}','x_{1ee}','x_{2ee}','x_{1rk}','x_{2rk}','x_{1ie}','x_{ie}')

% Plot accuracy
subplot(2,1,2)
semilogy(T,sum((solIEU - solRef').^2)./n_f)
legend('ee','rk','ie')

%% Task 3.6
min_h = -3;
max_h = 2;
n_points = 100;
mu = 1;
h_v = logspace(min_h,max_h,n_points);
accuracyh = zeros(3,n_points);
for i = 1:n_points
    [T, ref_sol] = ode15s(@(t,x) van_der_pol(t,x,mu),[0 h_v(i)],x0);
    
    % YOUR CODE HERE: 
    % ieu_sol = ...;
    % rk4_sol = ...;
    % eeu_sol = ...;
    
    accuracyh(1,i) = norm(eeu_sol - ref_sol(end,:)')/norm(ref_sol);
    accuracyh(2,i) = norm(rk4_sol - ref_sol(end,:)')/norm(ref_sol);
    accuracyh(3,i) = norm(ieu_sol - ref_sol(end,:)')/norm(ref_sol);
end

figure()
loglog(h_v,accuracyh)
xlabel('step-size')
ylabel('accuracy')
grid on
legend('ee','rk','ie')
ylim([1.0e-8,1.0e10]);