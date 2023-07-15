syms h(u,v)
% syms h(u,v,X,Y)

% X = rand(1,1);
% Y = rand(1,1);
syms X Y
X = sym('X', 'real');
Y = sym('Y', 'real');

h(u,v,X,Y) = (X/2)*u^2+(Y/2)*v^2;
Dh_u = diff(h, u);
Dh_uu = diff(h,u,2);
Dh_v = diff(h,v);
Dh_vv = diff(h,v,2);
Dh_uv= diff(Dh_u, v);

% disp(Dh_uv);


% Gaussian and Mean curvatures
K = (Dh_uu*Dh_vv-Dh_uv^2)/((1+Dh_u^2+Dh_v^2)^2);
H = ((1+Dh_v^2)*Dh_uu - 2*Dh_u*Dh_v*Dh_uv + (1+Dh_u^2)*Dh_vv)/(2*(1+Dh_u^2+Dh_v^2)^(3/2));

disp(K);
disp(H);

% Prinicpal curvatures
k1 = H + (H^2-K)^(1/2);
k2 = H - (H^2-K)^(1/2);

% We assume w.l.o.g that the principal directions are D1=(1,0)=u and D2=(0,1)=v
% this is true if X>Y, else it will be flipped
Dk1_1 = diff(k1, u);
disp(Dk1_1);
disp(Dk1_1(0,0, X, Y))
Dk1_2 = diff(k1, v);
Dk2_1 = diff(k2, u);
Dk2_2 = diff(k2, v);
Dk1_22 = diff(Dk1_2, v);
Dk2_11 = diff(Dk2_1, u);

% Calculate correlations
% disp(calculate_correlation_monte_carlo(k1,k2));
% N=5000;
% disp('corr(k1,k2');
% disp(calculate_correlation_monte_carlo(k1,k2,N));
% disp('corr(k1,Dk1_1');
% disp(calculate_correlation_monte_carlo(k1,Dk1_1,N));
% disp('corr(k1,Dk1_2');
% disp(calculate_correlation_monte_carlo(k1,Dk1_2,N));
% disp('corr(k1,Dk2_1');
% disp(calculate_correlation_monte_carlo(k1,Dk2_1,N));
% disp('corr(k1,Dk2_2');
% disp(calculate_correlation_monte_carlo(k1,Dk2_2,N));
% disp('corr(k1,Dk2_11');
% disp(calculate_correlation_monte_carlo(k1,Dk2_11,N));
% disp('corr(k1,Dk1_22');
% disp(calculate_correlation_monte_carlo(k1,Dk1_22,N));

% disp(calculate_correlation(k1,k2));
disp(calculate_correlation(k1, Dk1_1));
disp(calculate_correlation(k1,Dk1_1));
disp(calculate_correlation(k1, Dk1_2));
disp(calculate_correlation(k1, Dk2_1));
disp(calculate_correlation(k1,Dk2_2));
disp(calculate_correlation(k1,Dk1_22));
disp(calculate_correlation(k1,Dk2_11));



%  a= 0;
% b = 1;
% f_X = 1/(b-a);
% f_Y = 1/(b-a);
% % E_k1 = int(k1*f_X, X, a, b);
% % E_k1 = int(E_k1*f_Y, Y, a, b);
% % % disp(double(E_k1));
% % E_k2 = int(k2*f_X, X, a, b);
% % E_k2 = int(E_k2*f_Y, Y, a, b);
% 
% N = 1000;  % Number of samples
% 
% % Generate random samples for X and Y
% X_samples = a + (b - a) * rand(N, 1);
% Y_samples = a + (b - a) * rand(N, 1);
% k1_values = subs(k1, [u,v,X, Y], {0,0,X_samples, Y_samples});
% k2_values = subs(k2, [u,v,X, Y], {0,0,X_samples, Y_samples});
% 
% % Calculate the sample means
% E_k1 = mean(double(k1_values));
% E_k2 = mean(double(k2_values));
% 
% % Calculate the sample covariance
% cov_k1_k2 = mean((double(k1_values) - E_k1) .* (double(k2_values) - E_k2));
% 
% % Calculate the sample variances
% var_k1 = mean((double(k1_values) - E_k1).^2);
% var_k2 = mean((double(k2_values) - E_k2).^2);
% 
% % Calculate the correlation coefficient
% corr = cov_k1_k2 / sqrt(var_k1 * var_k2);

% Ampiric experiment
% X = 0.7;
% Y = 0.3;
% disp(k1(0,0,X,Y))
% disp(k2(0,0,X,Y))
% disp(Dk1_1(0,0, X, Y));
% disp(Dk1_2(0,0,X, Y));
% disp(Dk2_1(0,0,X,Y));
% disp(Dk2_2(0,0,X,Y));
% disp(Dk2_11(0,0,X,Y));
% disp(Dk1_22(0,0,X,Y));