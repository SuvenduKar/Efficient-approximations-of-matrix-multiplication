% This is first order SVD based approximation of A*B
function [comp,err,time]=ne_svdmul1(A,B,tol)
    AB=A*B;
    AB_fro=norm(AB,'fro');
    t_start=tic;
    [m,n]=size(A);
    comp=1;
    assert(size(A,2) == size(B,1), 'Inputs must be Multiplication compatiable.');
    [U1, Sigma1, V1] = randomized_partial_svd(A, comp);
    [U2, Sigma2, V2] = randomized_partial_svd(B, comp);
    
    a = min(1 * floor(log2(n)), n); 
    start=1;
    terminal=start+a-1;
    while (comp*a)<=n
        
        
        B_constructed=(U2.*Sigma2')*V2';
        delB=B-B_constructed;
        AB_constructed=((A*U2).*Sigma2')*V2';
        AdelB_constructed=(U1.*Sigma1')*(V1'*delB);
        

        err_time_start=tic;
        [err]=err_in_approximation(AB,AB_constructed,AdelB_constructed,AB_fro);
        
        err_time=toc(err_time_start);
        if err<tol
            time=toc(t_start)-err_time;
            return;
        end
        if (terminal+1)<=n

            start=terminal+1;
        else
            time=toc(t_start)-err_time;
            return;
        end
        if start+a-1<=n
            terminal=start+a-1;
        else
            terminal=n;
        end
        comp=comp+1;
        t_start=tic;
        [U1, Sigma1, V1] = randomized_partial_svd(A, comp);
        [U2, Sigma2, V2] = randomized_partial_svd(B, comp);
        
        
    end

end