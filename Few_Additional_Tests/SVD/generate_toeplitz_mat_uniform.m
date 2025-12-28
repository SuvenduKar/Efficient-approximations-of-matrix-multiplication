
% generate Toeplitz matrix with the entries from uniform (0,1) distribution
function M=generate_toeplitz_mat_uniform(order)
    c = rand(order, 1);    % First column (random)
    r = rand(1, order);    % First row (random)
    r(1) = c(1);        % Ensure top-left element matches
    M = toeplitz(c, r);
end