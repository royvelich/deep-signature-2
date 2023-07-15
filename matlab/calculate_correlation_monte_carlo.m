function [corr] = calculate_correlation_monte_carlo(f,g, N)
    a= 0;
    b = 1;
    f_X = 1/(b-a);
    f_Y = 1/(b-a);
    syms u v X Y
    % E_k1 = int(k1*f_X, X, a, b);
    % E_k1 = int(E_k1*f_Y, Y, a, b);
    % % disp(double(E_k1));
    % E_k2 = int(k2*f_X, X, a, b);
    % E_k2 = int(E_k2*f_Y, Y, a, b);
    
    % N = 10000;  % Number of samples
    
    % Generate random samples for X and Y
    X_samples = a + (b - a) * rand(N, 1);
    Y_samples = a + (b - a) * rand(N, 1);
    f_values = subs(f, [u,v,X, Y], {0,0,X_samples, Y_samples});
    g_values = subs(g, [u,v,X, Y], {0,0,X_samples, Y_samples});
    
    % Calculate the sample means
    E_f = mean(double(f_values));
    E_g = mean(double(g_values));
    
    % Calculate the sample covariance
    cov_f_g = mean((double(f_values) - E_f) .* (double(g_values) - E_g));
    
    % Calculate the sample variances
    var_f = mean((double(f_values) - E_f).^2);
    var_g = mean((double(g_values) - E_g).^2);
    
    % Calculate the correlation coefficient
    corr = cov_f_g / sqrt(var_f * var_g);

end