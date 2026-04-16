function [repeat,err] = low_rank_approx_algo(A, B, tol)
    [m1, n1] = size(A);
    [m2, n2] = size(B);
    AB=A*B;
    if n1 ~= m2
        error('Multiplication of A and B is not possible.');
    end

    AB_fro = norm(AB, 'fro');
    repeat = 1;
    c = min(repeat * floor(log2(n1)) + 1, n1);
    
    tic;  % Start timer
    % Compute initial probability distribution
    prob_list = zeros(1, n1);
    for i = 1:n1
        prob_list(i) = norm(A(:, i), 2) * norm(B(i, :), 2);
    end
    prob_list = prob_list / sum(prob_list);

    % Sample indices
    K = randsample(n1, c, true, prob_list);
    
    % Approximate product
    M = zeros(m1, n2);
    for i = 1:length(K)
        ki = K(i);
        M = M + (1 / (c * prob_list(ki))) * (A(:, ki) * B(ki, :));
    end

    err = norm(AB - M, 'fro') / AB_fro;

    % Repeat until error ≤ tolerance or max allowed c reached
    while err > tol && (repeat * floor(log2(n1)) + 1 <= n1)
        repeat = repeat + 1;
        c = min(repeat * floor(log2(n1)) + 1, n1);
        
        % Recompute probabilities
        prob_list = zeros(1, n1);
        for i = 1:n1
            prob_list(i) = norm(A(:, i), 2) * norm(B(i, :), 2);
        end
        prob_list = prob_list / sum(prob_list);
        
        % Resample and recompute
        K = randsample(n1, c, true, prob_list);
        M = zeros(m1, n2);
        for i = 1:length(K)
            ki = K(i);
            M = M + (1 / (c * prob_list(ki))) * (A(:, ki) * B(ki, :));
        end

        err = norm(AB - M, 'fro') / AB_fro;
    end
    elapsedTime = toc;

    %fprintf('DS 298 algorithm used %d components for matrices of size %d and gave error of %f\n', c, n1, err);
end
