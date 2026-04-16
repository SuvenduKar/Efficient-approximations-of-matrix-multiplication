function S = sym_rand(n)
    A = rand(n);
    S = 0.5*(A + A');  % symmetric
end