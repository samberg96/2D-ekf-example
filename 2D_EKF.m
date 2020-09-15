%% AER1513 Assignment 2 - 2D EKF
% Sam Weinberg
% 09/15/2020

% This code uses an EKF to fuse velocity readings with laser range-finder
% measurements to estimate the 2D position of a mobile robot. Please see
% 'AER1513_Assignment_Weinberg.pdf' and 'State Estimation for Robotics' by
% Barfoot for full derivations and notation.

clc
close all 
clear all

load dataset2

% Sampling period
T = 0.1;

% Data length
K = length(v);

% User defined Range Threshold
r_max = 1;

% Intialize matrices
state_hat = zeros(K,3);
x_hat = zeros(K,1);
y_hat = zeros(K,1);
theta_hat = zeros(K,1);
P_hat = zeros(3,3,K);

% Initial values
state_hat(1,:) = [10e100 10e100 0.1].';
x_hat(1) = state_hat(1,1);
y_hat(1) = state_hat(1,2);
theta_hat(1) = state_hat(1,3);
P_hat(:,:,1) = diag([1, 1, 0.1]);

for k = 2:1:K   
    
    % Define process noise matrix
    Q = [v_var*(T*cos(theta_hat(k - 1)))^2, v_var*(T^2*cos(theta_hat(k - 1))*sin(theta_hat(k - 1))), 0;
        v_var*(T^2*cos(theta_hat(k - 1))*sin(theta_hat(k - 1))), v_var*(T*sin(theta_hat(k - 1)))^2, 0;
        0, 0, om_var*T^2];
      
    % Jacobian motion model
    F = [1 0 -T*v(k)*sin(theta_hat(k - 1));
        0 1 T*v(k)*cos(theta_hat(k - 1));
        0 0 1];
    
    % Predictor
    P_check = F*P_hat(:,:,k - 1)*F.' + Q;
    x_check = x_hat(k - 1) + T*v(k)*cos(theta_hat(k - 1));
    y_check = y_hat(k - 1) + T*v(k)*sin(theta_hat(k - 1));
    theta_check = theta_hat(k - 1) + T*om(k);
    
    state_check = [x_check, y_check, theta_check].';
        
    % For each landmark
    cnt = 0;
    G = zeros(2,3);
    innov = zeros(2,1);
    R = zeros(2,2);
        
    for i = 1:1:17               
        if r(k,i) ~= 0 && r(k,i) < r_max
                      
            cnt = cnt + 1;
            
            % Jacobian observation model
            dg1_dx = -(l(i,1) - x_check - d*cos(theta_check))/((l(i,1) - x_check - d*cos(theta_check))^2 + (l(i,2) - y_check - d*sin(theta_check))^2)^(1/2);
            dg1_dy = -(l(i,2) - y_check - d*sin(theta_check))/((l(i,1) - x_check - d*cos(theta_check))^2 + (l(i,2) - y_check - d*sin(theta_check))^2)^(1/2);
            dg1_dtheta = d*(sin(theta_check)*(l(i,1) - x_check - d*cos(theta_check)) - cos(theta_check)*(l(i,2) - y_check - d*sin(theta_check)))/((l(i,1) - x_check - d*cos(theta_check))^2 + (l(i,2) - y_check - d*sin(theta_check))^2)^(1/2);
            
            dg2_dx = (l(i,2) - y_check - d*sin(theta_check))/((l(i,1) - x_check - d*cos(theta_check))^2 + (l(i,2) - y_check - d*sin(theta_check))^2);
            dg2_dy = -(l(i,1) - x_check - d*cos(theta_check))/((l(i,1) - x_check - d*cos(theta_check))^2 + (l(i,2) - y_check - d*sin(theta_check))^2);
            dg2_dtheta = -d*sin(theta_check)*(l(i,2) - y_check - d*sin(theta_check))/((l(i,1) - x_check - d*cos(theta_check))^2 + (l(i,2) - y_check - d*sin(theta_check))^2) - d*cos(theta_check)*(l(i,1) - x_check - d*cos(theta_check))/((l(i,1) - x_check - d*cos(theta_check))^2 + (l(i,2) - y_check - d*sin(theta_check))^2) - 1;
            
            G(2*cnt - 1:2*cnt, 1:3) = [dg1_dx dg1_dy dg1_dtheta;
            dg2_dx dg2_dy dg2_dtheta];
            
            % Define measurement noise matrix
            R(2*cnt - 1:2*cnt, 2*cnt - 1:2*cnt) = diag([r_var, b_var]);
                  
            % Calculate innovation
            r_pred = ((l(i,1) - x_check - d*cos(theta_check))^2 + (l(i,2) - y_check - d*sin(theta_check))^2)^(1/2);
            b_pred = atan2(l(i,2) - y_check - d*sin(theta_check), l(i,1) - x_check - d*cos(theta_check)) - theta_check;                        
            
            if b_pred > pi
                b_pred = b_pred - 2*pi;
            elseif b_pred < -pi
                b_pred = b_pred + 2*pi;
            end
            
            innov(2*cnt - 1:2*cnt, 1) = [r(k,i); b(k,i)] - [r_pred; b_pred]; 
        end           
    end
    
    % Kalman Gain
    K = P_check*G.'/(G*P_check*G.' + R);
    
    % Corrector
    if cnt == 0
        P_hat(:,:,k) = P_check;        
        state_hat(k,:) = state_check;
        x_hat(k) = state_hat(k,1);
        y_hat(k) = state_hat(k,2);
        theta_hat(k) = state_hat(k,3);
    else       
        P_hat(:,:,k) = (eye(3) - K*G)*P_check;        
        state_hat(k,:) = state_check + K*innov;
        x_hat(k) = state_hat(k,1);
        y_hat(k) = state_hat(k,2);
        theta_hat(k) = state_hat(k,3);
    end
    
    % Keep theta between -pi and pi
    theta_hat(k) = wrapToPi(theta_hat(k));
    
end

% Extract covariances
for n = 1:1:length(x_true)
    x_var(n) = P_hat(1,1,n);
    y_var(n) = P_hat(2,2,n);
    theta_var(n) = P_hat(3,3,n);
end

% Error Plots
figure(1)
plot(t, x_hat - x_true)
hold on
plot(t, 3*sqrt(x_var), '--')
hold on
plot(t, -3*sqrt(x_var), '--')
axis([0 max(t) -0.5 0.5])
xlabel("Time (s)")
ylabel("x_E_r_r_o_r (m)")
legend("Error","3\sigma Uncertainty Envelope", "-3\sigma Uncertainty Envelope")
hold off

figure(2)
plot(t, y_hat - y_true)
hold on
plot(t, 3*sqrt(y_var), '--')
hold on
plot(t, -3*sqrt(y_var), '--')
axis([0 max(t) -0.5 0.5])
xlabel("Time (s)")
ylabel("y_E_r_r_o_r (m)")
legend("Error","3\sigma Uncertainty Envelope", "-3\sigma Uncertainty Envelope")
hold off

figure(3)
plot(t, wrapToPi(theta_hat - th_true))
hold on
plot(t, 3*sqrt(theta_var), '--')
hold on
plot(t, -3*sqrt(theta_var), '--')
axis([0 max(t) -0.5 0.5])
xlabel("Time (s)")
ylabel("\theta_E_r_r_o_r (rad)")
legend("Error","3\sigma Uncertainty Envelope", "-3\sigma Uncertainty Envelope")
hold off

% Animation
h = figure(4);
axis tight manual
filename = 'assignment2.gif';
plot(l(:,1), l(:,2),'.k')
hold on
est = animatedline('Marker','o','LineStyle','None','Color','red');
tru = animatedline('Marker','o','LineStyle','None','Color','blue');
elip = animatedline('Marker','.','LineStyle','None','Color','red');
axis([-2 10 -3 3])

for m = 1:1:length(x_true)
    
    % Create uncertainty ellipse
    s = -2 * log(1 - 0.99); % 0.99 for 3*sigma

    [V, D] = eig(P_hat(1:2,1:2,m) * s);

    t = linspace(0, 2 * pi);
    a = (V * sqrt(D)) * [cos(t(:))'; sin(t(:))'];
    
    clearpoints(elip)
    clearpoints(est)
    clearpoints(tru)

    addpoints(elip, a(1, :) + x_hat(m), a(2, :) + y_hat(m));    
    addpoints(est,x_hat(m),y_hat(m))
    addpoints(tru,x_true(m),y_true(m))
    
    drawnow limitrate
    
    % Capture the plot as an image 
    frame = getframe(h); 
    im = frame2im(frame); 
    [imind,cm] = rgb2ind(im,256); 
    % Write to the GIF File 
    if m == 1 
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
    elseif mod(m,2) == 0
        imwrite(imind,cm,filename,'gif','WriteMode','append'); 
    end 

end