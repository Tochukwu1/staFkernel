clear all
clc
close all

global u_star_store
global v_hat_store
global v_star_store
global u_hat_store
global t_store


% Simulation Parameters
P.n = 2;
P.l = 3;
P.kc1 = 0.001;
P.kc2 = 0.25;
P.ka1 = 1.2;
P.ka2 = 0.01;
P.Beta = 0.003;
P.gamma1 = 0.005;
P.gamma2 = 1;
P.R = eye(1);
P.N = 1;
P.u_star_store = [];
P.v_star_store = [];
P.u_hat_store = [];
P.t_store = [];

% Initial Conditions
gamma_t0 = 500*eye(3);
wc_hat0 = 0.4*ones(3,1);
wa_hat0 = 0.7*wc_hat0;
tspan = [0 10];
x0 = [-1 1]';
x0 = [x0;wa_hat0;wc_hat0;gamma_t0(:)];

options = odeset('OutputFcn',@odeplot, 'OutputSel', 1:2);
% Simulation
[t,x] = ode45(@(t,x) closeLoopDynamics(t,x, P), tspan, x0, options);


% Plots
p = plot(t,x(:, 1:2));
mrk1={'s','v','o','*','^'};
mrk=(mrk1(1,1:size(p,1)))';
set(p,{'marker'},mrk,{'markerfacecolor'},get(p,'Color'),'markersize', 5);
title('State Trajectory', 'FontSize',20,'Interpreter','latex')
set(gca,'FontSize',20)
xlabel('$Time (s)$','FontSize',20,'Interpreter','latex')
ylabel('$x(t)$','FontSize',20,'Interpreter','latex')
legend('x_1','x_2');
grid on

figure
p1 = plot(t_store, u_star_store);
set(p1,'Color',get(p(1),'Color'),'LineWidth', 2)
hold on 
p2 = plot(t_store, u_hat_store);
set(p2,'Color',get(p(2),'Color'),'LineWidth', 2)
title('Optimal Control Estimation', 'FontSize',20,'Interpreter','latex')
set(gca,'FontSize',20)
xlabel('$Time (s)$','FontSize',20,'Interpreter','latex')
ylabel('$u(t)$','FontSize',20,'Interpreter','latex')
legend('$u^*(x(t))$','$\hat{u}(x(t), \hat{W}_{a}(t))$', 'Interpreter','latex');
% xlim([0, 5])
% ylim([-2, 0.5])
grid on

figure
p3 = plot(t, x(:, 3:5));
mrk3=(mrk1(1,1:size(p3,1)))';
set(p3,{'marker'},mrk3,{'markerfacecolor'},get(p3,'Color'),'markersize', 5);
title('Policy Weights', 'FontSize',20,'Interpreter','latex')
set(gca,'FontSize',20)
xlabel('$Time (s)$','FontSize',20,'Interpreter','latex')
ylabel('$\hat{W}_{a}(t)$','FontSize',20,'Interpreter','latex')
legend('$\hat{W}_{a,1}(t)$','$\hat{W}_{a,2}(t)$', '$\hat{W}_{a,3}(t)$', 'Interpreter','latex');
grid on

figure
p4 = plot(t, x(:, 6:8));
mrk4=(mrk1(1,1:size(p3,1)))';
set(p4,{'marker'},mrk3,{'markerfacecolor'},get(p4,'Color'),'markersize', 5);
title('Value Function Weights', 'FontSize',20,'Interpreter','latex')
set(gca,'FontSize',20)
xlabel('$Time (s)$','FontSize',20,'Interpreter','latex')
ylabel('$\hat{W}_{c}(t)$','FontSize',20,'Interpreter','latex')
legend('$\hat{W}_{c,1}(t)$','$\hat{W}_{c,2}(t)$', '$\hat{W}_{c,3}(t)$', 'Interpreter','latex');
grid on


figure
v_error = v_star_store-v_hat_store;
plot(t_store, v_error)
title('Value Function Estimation Error', 'FontSize',20,'Interpreter','latex')
set(gca,'FontSize',20)
xlabel('$Time (s)$','FontSize',20,'Interpreter','latex')
ylabel('${V^*}(x(t))-\hat{V}(x(t), \hat{W}_{c,1}(t))$','FontSize',20,'Interpreter','latex')
% ylim([-8, 2])
grid on

% Functions

function [ydot] = closeLoopDynamics(t, y, P)
global u_hat_store
global u_star_store
global v_hat_store
global v_star_store
global t_store

x = y(1:P.n,:);
Wc_hat = y(P.n+P.l+1:P.n+P.l+P.l, :); % row 6-8
[u, sigma, Wa_hat_dot,Wc_hat_dot,Gamma_dot] = updateLawForActorCriticWeights(y, P);

u_star = optimalControl(t, x, P);
u_hat = u;

v_star = optimalValueFunction(t, x);

v_hat = Wc_hat'*sigma;
    
if t == 0
    u_hat_store = u_hat;
    u_star_store = u_star;
    v_hat_store = v_hat;
    v_star_store = v_star;
    t_store = t;
else
    u_hat_store = [u_hat_store; u_hat];
    u_star_store = [u_star_store; u_star];
    v_hat_store = [v_hat_store; v_hat];
    v_star_store = [v_star_store; v_star];
    t_store = [t_store; t];
end

xdot = dynamics(t, x,u_star, P);

ydot = [xdot;Wa_hat_dot;Wc_hat_dot;Gamma_dot(:)];
end

function xdot = dynamics(t, x, u, P)
    x1 = x(1);
    x2 = x(2);
    f_x = [-x1+x2; -1/2*x1-1/2*x2*(1-(cos(2*x1)+2)^2)];
    g_x = [0; cos(2*x1)+2];

    xdot = f_x+(g_x*u);
end

function u_star = optimalControl(t, x, P)
    x1 = x(1);
    x2 = x(2);
    u_star = -(cos(2*x1)+2)*x2;
end


function [sigma, grad_sigma_x] = basis(x, D, P)

    x = x(1:2,:);
    L = size(D, 2);
    sigma = zeros(L,1);
    grad_sigma_x = zeros(L, length(x));
    c = zeros(2,L);


    for i=1:size(D, 2)
        c(:,i) = x + D(:,i);
        sigma(i,1) = exp(x'*c(:,i))-1;
        grad_sigma_x(i,:) = c(:,i)*exp(x'*c(:,i));
    end
end

function f_x = fx(x)
    x1 = x(1);
    x2 = x(2);
    f_x = [-x1+x2; -1/2*x1-1/2*x2*(1-(cos(2*x1)+2)^2)];
end

function g_x = gx(x)
    g_x = [0; cos(2*x(1))+2];
end


function [u, sigma, Wa_hat_dot,Wc_hat_dot,Gamma_dot] = updateLawForActorCriticWeights(y, P)


   x = y(1:P.n, :); % row 1-2
   Wa_hat = y(P.n+1:P.n+P.l, :); % row 3-5
   Wc_hat = y(P.n+P.l+1:P.n+P.l+P.l, :); % row 6-8
   Gamma = reshape(y(P.n+P.l+P.l+1:P.n+P.l+P.l+P.l^2), P.l, P.l); % row 9-17
   
   summ_wc_r = zeros(size(Wc_hat));
   summ_gamma_r  = zeros(size(Wa_hat,1),size(Wa_hat,1));
   summ_wa_r  = zeros(size(Wa_hat));
   
   nu =((x'*x)+0.01)/(1+P.gamma2*(x'*x));
   ai_1 = meshgrid(linspace(-2.1*nu,2.1*nu,1), linspace(-2.1*nu,2.1*nu,1));
   ai_2 = ai_1';
   ai = [ai_1(:) ai_2(:)];
   %   ai = unifrnd(-2.1*nu,2.1*nu,2,1);
   d1 = (0.7*nu)*[0,1]';
   d2 = (0.7*nu)*[0.87, -0.5]';
   d3 = (0.7*nu)*[-0.87, -0.5]';
   D = [d1 d2 d3];
   xi = x + ai';

   for i=1:P.N
       xi = xi(:, 1);
       [~, grad_sigma_y] = basis(xi,D,P);
       ui = -1/2*P.R^-1*gx(xi)'*grad_sigma_y'*Wa_hat;
       G_sigmai = grad_sigma_y*gx(xi)*(P.R^-1)*gx(xi)'*grad_sigma_y';
       wi = grad_sigma_y*fx(xi) + grad_sigma_y*gx(xi)*ui;
       pi = 1 +(P.gamma1*(wi'*wi));
       gradv_hat_i = Wc_hat'*grad_sigma_y;
       ri = xi'*xi+ui^2;
       delta_ti = ri + gradv_hat_i*(fx(xi)+gx(xi)*ui);
       summ_wc_r = summ_wc_r +(wi/pi)*delta_ti;
       summ_gamma_r = summ_gamma_r + (wi*wi'/pi^2)*Gamma;
       summ_wa_r = summ_wa_r + ((P.kc2*G_sigmai'*Wa_hat*wi')*Wc_hat/(4*P.N*pi));
   end
   
    [sigma, grad_sigma_x] = basis(x,D,P);
    u = -1/2*P.R^-1*gx(x)'*grad_sigma_x'*Wa_hat;
    w = grad_sigma_x*fx(x) + grad_sigma_x*gx(x)*u;
    p = 1 +(P.gamma1*(w'*w));
    G_sigma = grad_sigma_x*gx(x)*P.R^-1*gx(x)'*grad_sigma_x';
    gradv_hat = Wc_hat'*grad_sigma_x;
    r = x'*x+u^2;
    delta_t = r + gradv_hat*(fx(x)+gx(x)*u);
    
    Wc_hat_dot = (-P.kc1*Gamma*(w/p)*delta_t)-(P.kc2/P.N)*Gamma*summ_wc_r;
   
   
   Gamma_dot = P.Beta*Gamma-(P.kc1*Gamma*(w*w'/p^2)*Gamma)-(P.kc2/P.N)*...
        Gamma*summ_gamma_r;
    
    Wa_hat_dot = -P.ka1*(Wa_hat-Wc_hat)-P.ka2*Wa_hat+((P.kc1*G_sigma'*Wa_hat...
    *w')/(4*p))*Wc_hat + summ_wa_r;
   
end


function v_star = optimalValueFunction(t, x)
    x1 = x(1);
    x2 = x(2);
    v_star = 1/2*x1^2 + x2^2;
end


