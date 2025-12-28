% This is zeroth order SVD based approximation of multiplication A*B

function [comp,err]=ne_svdmul0(A,B,tol)
    AB=A*B;
    [m,n]=size(A);
    comp=1;
    assert(size(A,2) == size(B,1), 'Inputs must be Multiplication compatiable.');
    [U1, Sigma1, V1] = randomized_partial_svd(A, comp);
    [U2, Sigma2, V2] = randomized_partial_svd(B, comp);
    [~, indicesA] = sort(Sigma1, 'descend');
    [~, indicesB] = sort(Sigma2, 'descend');
    A_constructed = zeros(m,n);
    B_constructed = zeros(m,n);

    
    a = min(1 * floor(log2(n)), n); 
    start=1;
    terminal=start+a-1;
    while (comp*a)<=n
        
        for k=1:length(Sigma1)
            k1=indicesA(k);
            k2=indicesB(k);
            
            A_constructed=A_constructed+Sigma1(k1)*U1(:,k1)*V1(:,k1)';
    
            B_constructed=B_constructed+Sigma2(k2)*U2(:,k2)*V2(:,k2)';
        end
        err=norm(AB-A_constructed*B_constructed,'fro')/norm(AB,'fro'); % This implementation of the multiplication can be made optimal wrt computation count as per the corresponding method in the paper
        if err<tol
            return;
        end
        if (terminal+1)<=n

            start=terminal+1;
        else
            return;
        end
        if start+a-1<=n
            terminal=start+a-1;
        else
            terminal=n;
        end
        comp=comp+1;
        [U1, Sigma1, V1] = randomized_partial_svd(A, comp);
        [U2, Sigma2, V2] = randomized_partial_svd(B, comp);
        [~, indicesA] = sort(Sigma1, 'descend');
        [~, indicesB] = sort(Sigma2, 'descend');
        A_constructed = zeros(m,n);
        B_constructed = zeros(m,n);
    end

end

