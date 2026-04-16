function [indices] = top_indices(R)
    % R is a cell array where each element is a column
    % Each R{i} is assumed to have first column
    % Returns top indices based on the squared 2-norm of the first column

    n = size(R,3);
    

    P = zeros(1, n);
    for i = 1:n
        term=R(:,:,i);
        P(i) = norm(term, 2)^2;
    end

    % Get indices of top 'a' values in P, sorted by value then index
    [~, indices] = sort(P, 'descend');
    

    % %Return indices in ascending order
    %indices = sort(top_a_indices);
end
