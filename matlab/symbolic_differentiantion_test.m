% Syms
syms u v L N A B C

% Define the function h
h = L * (u ^ 3) + N * (v ^ 3) + A * u * v;
x = [u; v; h];

% Define the first and second derivatives
h_u = diff(h, u);
h_v = diff(h, v);
h_uu = diff(h_u, u);
h_uv = diff(h_u, v);
h_vv = diff(h_v, v);

% Define the Gaussian curvature K and the mean curvature H
K = (h_uu * h_vv - h_uv ^ 2) / ((1 + h_u ^ 2 + h_v ^ 2) ^ 2);
H = ((1 + h_v ^ 2) * h_uu - 2 * h_u * h_v * h_uv + (1 + h_u ^ 2) * h_vv) / (2 * ((1 + h_u ^ 2 + h_v ^ 2) ^ (3/2)));
k1 = H + sqrt(H ^ 2 - K);
k2 = H - sqrt(H ^ 2 - K);

% Define the Weingarten matrix
x_u = diff(x, u);
x_v = diff(x, v);
n = cross(x_u, x_v);
n = n / norm(n);
x_uu = diff(x_u, u);
x_vv = diff(x_v, v);
x_uv = diff(x_u, v);
w = -[dot(x_uu, n), dot(x_uv, n); dot(x_uv, n), dot(x_vv, n)];

% disp(sqrt(1 + h_u ^ 2 + h_v ^ 2));
w2 = [h_uu / sqrt(1 + h_u ^ 2 + h_v ^ 2), h_uv / sqrt(1 + h_u ^ 2 + h_v ^ 2); h_uv / sqrt(1 + h_u ^ 2 + h_v ^ 2), h_vv / sqrt(1 + h_u ^ 2 + h_v ^ 2)];

% Find the eigen vectors
[V, D] = eig(w2);
d1 = V(:, 1);
d2 = V(:, 2);

% Normalize d1 and d2
d1_normalized = d1 / norm(d1);
d2_normalized = d2 / norm(d2);

% define the point
point_u = 0.7;
point_v = 0.9;
point_L = 1;
point_N = 1;
point_A = -0.2;

% evaluate the expressions at the point
d1_val = subs(d1_normalized, [u, v, L, N, A], [point_u, point_v, point_L, point_N, point_A]);
d2_val = subs(d2_normalized, [u, v, L, N, A], [point_u, point_v, point_L, point_N, point_A]);

% % Convert symbolic objects to floats
% d1_val_float = double(d1_val);
% d2_val_float = double(d2_val);
% 
% % Compute the L2 norm of the vectors
% norm_d1 = norm(d1_val_float);
% norm_d2 = norm(d2_val_float);
% 
% % Normalize the vectors
% d1_normalized = d1_val_float / norm_d1;
% d2_normalized = d2_val_float / norm_d2;

fprintf('d1_normalized: [%.16f, %.16f]\n', d1_val);
fprintf('d2_normalized: [%.16f, %.16f]\n', d2_val);
