function A = type3mat(order)
    % Create diagonal matrix S with linearly decreasing values
    S = diag((order:-1:1) / order);

    % Generate first random orthogonal matrix Q1
    M1 = randn(order);
    [Q1, R1] = qr(M1);
    d = sign(diag(R1));
    d(d == 0) = 1;        % Avoid division by zero
    Q1 = Q1 .* d';        % Adjust signs across columns
    R1 = R1 .* d;
    
    % Generate second random orthogonal matrix Q2
    M2 = randn(order);
    [Q2, R2] = qr(M2);
    d = sign(diag(R2));
    d(d == 0) = 1;
    Q2 = Q2 .* d';
    R2 = R2 .* d;

    % Construct the final matrix
    A = Q1 * S * Q2';
end
