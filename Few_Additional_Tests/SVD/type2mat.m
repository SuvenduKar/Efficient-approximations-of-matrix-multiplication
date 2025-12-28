function A = type2mat(order)
    % Create diagonal matrix S with exponentially decaying entries
    S = diag(exp(-(0:order-1)/10));

    % Generate first random orthogonal matrix Q1
    M1 = randn(order);
    [Q1, R1] = qr(M1);
    d = sign(diag(R1));
    d(d == 0) = 1;      % Avoid division by zero
    Q1 = Q1 .* d';      % Correct signs
    R1 = R1 .* d;

    % Generate second random orthogonal matrix Q2
    M2 = randn(order);
    [Q2, R2] = qr(M2);
    d = sign(diag(R2));
    d(d == 0) = 1;
    Q2 = Q2 .* d';
    R2 = R2 .* d;

    % Construct matrix M
    M = Q1 * S * Q2';

    % Generate uniform noise matrix U and scale it
    U = rand(order);  % Uniform in [0,1]
    scale = 0.5 * norm(M, 'fro') / norm(U, 'fro');
    U = scale * U;

    % Final matrix
    A = M + U;
end
