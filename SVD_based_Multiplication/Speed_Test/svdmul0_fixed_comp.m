% This is zeroth order SVD based approximation of multiplication A*B 
% Here we will check the error of this approximation method given a fixed
% number of log2(n) components.

function [err]=svdmul0_fixed_comp(A,B)
    AB=A*B;
    [m,n]=size(A);
    comp=5; % Number of log2(n) components to be taken into account 
    assert(size(A,2) == size(B,1), 'Inputs must be Multiplication compatiable.');
    [U1, Sigma1, V1] = randomized_partial_svd(A, comp);
    [U2, Sigma2, V2] = randomized_partial_svd(B, comp);
    [~, indicesA] = sort(Sigma1, 'descend');
    [~, indicesB] = sort(Sigma2, 'descend');
    A_constructed = zeros(m,n);
    B_constructed = zeros(m,n);

    
    a = min(1 * floor(log2(n)), n); 
   
    
        
    for k=1:length(Sigma1)
        k1=indicesA(k);
        k2=indicesB(k);
        
        A_constructed=A_constructed+Sigma1(k1)*U1(:,k1)*V1(:,k1)';

        B_constructed=B_constructed+Sigma2(k2)*U2(:,k2)*V2(:,k2)';
    end
    err=norm(AB-A_constructed*B_constructed,'fro')/norm(AB,'fro'); % This implementation of the multiplication can be made optimal wrt computation count as per the corresponding method in the paper
        
        
    end



