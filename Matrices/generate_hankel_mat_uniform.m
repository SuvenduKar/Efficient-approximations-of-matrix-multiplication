function M=generate_hankel_mat_uniform(order)
    first_col = rand(order, 1);
    last_row = rand(1, order);
    last_row(1) = first_col(end);  % Ensure overlap element is consistent
     
    % % Create the Hankel matrix
    M = hankel(first_col, last_row);
end