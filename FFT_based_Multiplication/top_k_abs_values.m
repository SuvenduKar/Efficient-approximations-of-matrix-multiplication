function N = top_k_abs_values(M, k)
    % Create a zero matrix of the same size as M
    [m, n] = size(M);
    N = zeros(m, n);
    
    % Process each row
    for i = 1:m
        row = M(i, :);
        
        % Get indices of the top k absolute values
        [~, idx] = sort(abs(row), 'descend');
        top_k_indices = idx(1:k);
        
        % Copy the top k values to N at the same positions
        N(i, top_k_indices) = M(i, top_k_indices);
    end
end
