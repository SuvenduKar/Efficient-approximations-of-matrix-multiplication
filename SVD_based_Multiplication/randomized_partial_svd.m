function [U, Sigma, V] = randomized_partial_svd(A, repeat)
    [m, n] = size(A);
    c = min(repeat * floor(log2(n)) + 1, n);

    phi = rand(n, c);       % Random test matrix
    Y = A * phi;            % Project A to lower dimension

    [Q, R] = qr(Y, 0);      % Economy QR decomposition

    B = Q' * A;             % Project A onto Q
    [U_tilde, S, V] = svd(B, 'econ');  % Thin SVD of reduced matrix
    U = Q * U_tilde;        % Lift U back to original space

    Sigma = diag(S);        % Convert singular values to vector
end
