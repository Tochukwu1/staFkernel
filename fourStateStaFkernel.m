clear all
clc
close all

% Simulation Parameters
P.kc1 = 0.001;
P.kc2 = 1;
P.ka1 = 0.05;
P.ka2 = 0.001;
P.Beta = 0.01;
P.gamma1 = 0.1;
P.gamma2 = 1;
P.k = 500;
P.sigma_theta = eye(3);
P.sigma0 = 50*eye(5);
P.k_theta = 20;
P.R = eye(1);
P.N = 1;
global thetaT; 
thetaT = [-1 1 0;-0.5 0 -0.5];
P.D = 0.01;
P.x0 = [0 0].';
x_0 = [0 0].';
x_hat = [0 0].';
P.gamma = 100;
gamma_t0 = 50*eye(5);
wc_hat0 = 0.025*ones(5,1);
wa_hat0 = 0.025*ones(5,1);
xd0 = [0 1].';
theta0 = zeros(3,2);
Y_x0 = zeros(12,1);
Yf0 = zeros(36,1);
Gs0 = zeros(2,1);
Xf0 = zeros(6,1);

options = odeset('OutputFcn',@odeplot, 'OutputSel', 40:43);

x0 = [x_0;xd0;wa_hat0;wc_hat0;gamma_t0(:);theta0(:);Y_x0;Yf0(:);Gs0(:);Xf0];
tspan = 0:0.01:10;

% Simulation
[t,x] = ode45(@(t,x) closeLoopDynamics(t,x, P), tspan, x0,options);
 x_ = x(:,1:2);
 xd = x(:,3:4);
 e = x_-xd;

% Plots
plot(t, e);
legend('e1', 'e2')
title('Graph of Error', 'Interpreter','Latex')
grid on


figure
plot(t, x(:, 40:45))
yline(thetaT(1,1), '--','Color', 'cyan')
yline(thetaT(1,2), '--','Color', 'green')
yline(thetaT(1,3), '--','Color', 'magenta')
yline(thetaT(2,1), '--','Color', 'blue')
yline(thetaT(2,2), '--','Color', 'black')
yline(thetaT(2,3), '--','Color', 'yellow')
title('Graph of $\theta$', 'Interpreter','Latex')
xlabel('$t$ (s)','Interpreter','Latex')
ylabel('$\theta$ (rad)','Interpreter','Latex')
legend('$\theta_1$','$\theta_2$','$\theta_3$','$\theta_4$','$\theta_5$','$\theta_6$','Interpreter','Latex');
grid on 

figure
plot(t, x(:, 5:9));
title('Policy Weights', 'FontSize',20,'Interpreter','latex')
xlabel('$Time (s)$','FontSize',20,'Interpreter','latex')
ylabel('$\hat{W}_{a}(t)$','FontSize',20,'Interpreter','latex')
grid on

figure
p4 = plot(t, x(:, 10:14));
title('Value Function Weights', 'FontSize',20,'Interpreter','latex')
xlabel('$Time (s)$','FontSize',20,'Interpreter','latex')
ylabel('$\hat{W}_{c}(t)$','FontSize',20,'Interpreter','latex')
grid on


function [ydot] = closeLoopDynamics(t, y, P)
   x = y(1:2,:);
   theta_hat = y(40:45, :);
   Y_x = y(46:57, :);
   Y_x = reshape(Y_x, 2, 6);
   Yf= y(58:93, :);
   Yf = reshape(Yf, 6, 6);
   Xf =  y(96:101, :);

   Gs = y(94:95, :);

[u, wa_hat_dot, wc_hat_dot, gamma_t_dot] = systemIdentification(t, y, P);

    xdot = dynamics(t, x, u, P);
    
    minEig = 0.01; % Threshold to turn off filters
    update = 1;
    
    if norm(Yf) > 500
        update = 0;
    end
     
    update1 = 1;
    
    if min(eig(Yf)) > minEig
        update1 = 0;
    end
      Y_dot = Y(x)*update;    


      Yf_dot = (Y_x.'*Y_x)*update;
      Gs_dot = [0; (cos(2*x(1))+2)]*u*update;
      Xf_dot = Y_x.'*(x-P.x0-Gs)*update;
      
      xd_dot = desiredTrajectory(t, y);
      
      
      if update1 == 0
         theta_hat_dot = zeros(6,1);
      else
         theta_hat_dot = P.k_theta*P.gamma*(Xf-Yf*theta_hat);
      end
  
%        theta_hat_dot = zeros(6,1);

    ydot = [xdot;xd_dot;wa_hat_dot;wc_hat_dot;gamma_t_dot;theta_hat_dot;...
        Y_dot(:);Yf_dot(:);Gs_dot;Xf_dot(:)];
end


% Functions
function [u, wa_hat_dot, wc_hat_dot, gamma_t_dot] = systemIdentification(t, y, P)

    x = y(1:2,:);
    xd = y(3:4,:);
    e = x-xd;
    zeta = [e.' xd.'].';
    theta_hat =  y(40:45,:);
    theta_hat = reshape(theta_hat.',3,2,[]);
    wa_hat = y(5:9,:);
    wc_hat = y(10:14,:);
    gamma_t = y(15:39, :);
    gamma_t = reshape(gamma_t.',5,5,[]);
    
    for i=1:P.N
        nu = 1;
        ai = unifrnd(-2.1, 2.1, 4, 1);
        zeta_i = zeta + ai;
        ei=zeta_i(1:2,:);
        xdi=zeta_i(3:4,:);
        xi= ei+xdi;
        hdi = hd(xi);
        gi= gx(xi);
        g_plus_di= gPlus(xdi) ;
        F_1i = [-hdi+gi*g_plus_di*hdi;hdi];
        sigma_theta_xi = sigma_theta(xi);
        sigma_theta_di = sigma_theta(xdi);
        G_zeta = G(zeta_i);
       [grad_sigma_zeta_i] = basis(zeta_i, P);        
        mui= controlEstimate(zeta_i, y, P);
        F_thetai=[theta_hat'*sigma_theta_xi-gi*g_plus_di*theta_hat'*sigma_theta_di;zeros(2,1)];
        wi = grad_sigma_zeta_i*(F_thetai+F_1i+G_zeta*mui);
        pi=(1+nu*(wi'*gamma_t*wi));
        G_sigmai = (grad_sigma_zeta_i*G(zeta_i))*(P.R^-1)*(G(zeta_i)).'*(grad_sigma_zeta_i).';
       
        delta_ti = delta(t, y, zeta_i, mui, wc_hat, P);

        summationWcR = (wi/pi)*delta_ti;
        summationGammaR = (wi*wi'/pi.^2);
        summationWaR = ((P.kc2*G_sigmai'*wa_hat*wi.')/(4*P.N*pi));
             
    end
    
    mu = controlEstimate(zeta, y, P);
    u = control(t, y, mu);
    [grad_sigma_zeta] = basis(zeta, P);

    G_sigma = (grad_sigma_zeta*G(zeta))*(P.R^-1)*(G(zeta)).'*(grad_sigma_zeta).';
    F_1 = F1(y);
    F_theta = FTheta(y);
    w = grad_sigma_zeta*(F_theta + F_1+ G(zeta)*mu);
    
    p=(1+nu*(w'*gamma_t*w));
    
    gamma_t_dot = P.Beta*gamma_t-(P.kc1*gamma_t*(w*w.'/p.^2)*gamma_t)-...
        (P.kc2/P.N)*gamma_t*summationGammaR*gamma_t;
    gamma_t_dot = gamma_t_dot(:);

    delta_t = delta(t, y, zeta, mu, wc_hat, P);
 
    wc_hat_dot = (-P.kc1*gamma_t*(w/p)*delta_t)-(P.kc2/P.N)*gamma_t*summationWcR;

    wa_hat_dot = -P.ka1*(wa_hat-wc_hat)-P.ka2*wa_hat+((P.kc1*G_sigma.'*wa_hat...
        *w.')/(4*p))*wc_hat + summationWaR*wc_hat;
end

function F_theta = FTheta(y)
    x = y(1:2,:);
    xd = y(3:4,:);
   theta_hat =  y(40:45,:);
   theta_hat = reshape(theta_hat.',3,2,[]);
    g_plus_d = gPlus(x) ; 
    sigma_theta_x = sigma_theta(x);
    sigma_theta_d = sigma_theta(xd);
    F_theta = [theta_hat'*sigma_theta_x-gx(x)*g_plus_d*theta_hat'*sigma_theta_d;zeros(2,1)];
end

function F_1 = F1(y)
    x = y(1:2,:);
    xd = y(3:4,:);
    h_d = hd(xd);

    g_plus_d = gPlus(x) ; 
    F_1 = [-h_d+gx(x)*g_plus_d*h_d; h_d];
end

function x_dot = dynamics(t, x, u, P)


    f_x=[-x(1)+x(2) ;-0.5*x(1)-0.5*x(2)*(1-(cos(2*x(1))+2)^2)];
    g_x = [0; (cos(2*x(1))+2)];

    x_dot = f_x+(g_x*u);
end


function [grad_sigma_zeta] = basis(zeta, P)
        nu = 1;
        d1=nu.*[0;0;1/sqrt(3);1];
        d2=nu.*[0;0;1/sqrt(3);-1];
        d3=nu.*[0;0;-2/sqrt(3);0];
        d4=nu.*[0;0;1/sqrt(10);1/sqrt(6)];
        d5=nu.*[0;0;-2*sqrt(2/5);0];
        D = [d1 d2 d3 d4 d5];

    grad_sigma_zeta=[(2*zeta+D(:,1))'*exp(zeta'*(zeta+D(:,1)));
           (2*zeta+D(:,2))'*exp(zeta'*(zeta+D(:,2)));
           (2*zeta+D(:,3))'*exp(zeta'*(zeta+D(:,3)));
           (2*zeta+D(:,4))'*exp(zeta'*(zeta+D(:,4)));
           (2*zeta+D(:,5))'*exp(zeta'*(zeta+D(:,5)))];
end
 

function f_x = fx(x)
     f_x=[-x(1)+x(2) ;-0.5*x(1)-0.5*x(2)*(1-(cos(2*x(1))+2)^2)];
end

function g_x = gx(x)
    g_x = [0; cos(2*x(1))+2];
end

function mu = controlEstimate(zeta, y, P)
    wa_hat = y(5:9, :);
    [grad_sigma_zeta] = basis(zeta, P);

    mu = -1/2*(P.R^-1)*G(zeta).'*grad_sigma_zeta.'*wa_hat;
end

function delta = delta(t, y, zeta, mu, wc_hat, P)
    e = zeta(1:2,:);
   
    [grad_sigma_zeta] =  basis(zeta, P);
    gradv_hat = wc_hat.'*grad_sigma_zeta;
    r = e.'*diag([10,10])*e + mu.^2;
    F_theta = FTheta(y);
    F_1 = F1(y);
    delta = r + gradv_hat*(F_theta+F_1+G(zeta)*mu);
end

function g_plus = gPlus(x)  
     g_plus =[0 1/(cos(2*x(1))+2)];
end

function xd_dot = desiredTrajectory(t, y)
    xd = y(3:4);
    xd_dot = [-1 1; -2 1]*xd;
end

function u = control(t, y, mu)
      ud = steadyStateControlPolicy(t, y);
      u = ud + mu;
end

function sigma_theta = sigma_theta(x)
    sigma_theta = [x(1);x(2);x(2)*(1-(cos(2*x(1))+2)^2)];
end

function G_zeta = G(zeta)
    G_zeta = [gx(zeta(1:2, :)+ zeta(3:4, :)).', zeros(1, 2)].';
end

% function F_zeta = F(t, zeta, y)
% 
%     xd = zeta(3:4, :);
%     hd = [-1 1; -2 1]*xd;
%     ud = steadyStateControlPolicy(t,y);
%     F_zeta = [fx(zeta(1:2, :)+ zeta(3:4, :)).'-hd.'+ud.'*...
%         gx(zeta(1:2, :)+ zeta(3:4, :)).', hd.'].';
% end

function ud = steadyStateControlPolicy(t, y) 
   xd = y(3:4, :);
    theta_hat =  y(40:45,:);
   theta_hat = reshape(theta_hat.',3,2,[]);
% theta_hat = [-1 1 0;-0.5 0 -0.5].';
   ud = gPlus(xd)*(hd(xd)-theta_hat'*sigma_theta(xd));
%    ud  = pinv(gx(xd))*([-1 1; -2 1]*xd - theta_hat.'*sigma_theta(xd));
end

function h_d = hd(xd)
    h_d = [-1 1; -2 1]*xd;
end 

function Y = Y(x)
%Y
%   
%    Regressor Matrix
%   
    Y = [x(1)  x(2) 0  0  0 0;
         0   0  0 x(1) 0  x(2)*(1-(cos(2*x(1))+2)^2)];
    
end



