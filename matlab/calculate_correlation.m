function [corr] = calculate_correlation(f,g)
    syms X Y u v
    a = 0;  % Lower bound of the uniform distribution
    b = 1;  % Upper bound of the uniform distribution

    % Define the joint probability density function (PDF) of X and Y as 1/(b-a)^2
    f_XY = 1 / ((b - a)^2);

    % Calculate the expected values of f and g
    E_f = int(int(f(0, 0, X, Y) * f_XY, X, a, b), Y, a, b);
    E_g = int(int(g(0, 0, X, Y) * f_XY, X, a, b), Y, a, b);

    % Define the covariance and variances
    cov_fg = E_f * E_g - int(int(f(0, 0, X, Y) * g(0, 0, X, Y) * f_XY, X, a, b), Y, a, b);
    var_f = E_f^2 - int(int(f(0, 0, X, Y)^2 * f_XY, X, a, b), Y, a, b);
    var_g = E_g^2 - int(int(g(0, 0, X, Y)^2 * f_XY, X, a, b), Y, a, b);

    % Calculate the correlation coefficient
    corr = cov_fg / sqrt(var_f * var_g);
end