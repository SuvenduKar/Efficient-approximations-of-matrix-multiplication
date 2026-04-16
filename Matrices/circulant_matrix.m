
% This function is to create a circulant matrix given its first column.
function C = circulant_matrix(c)
    n = length(c);
    C = zeros(n);  % initialize square matrix

    for i = 1:n
        C(:, i) = circshift(c, i - 1);  % shift down by (i-1)
    end
end
