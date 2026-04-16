% This is first order SVD based approximation of A*B
function [comp,err,A_constructed,B_constructed,M]=svdmul1(A,B,tol)
    AB=A*B;
    AB_fro=norm(AB,'fro');
    
    [m,n]=size(A);
    comp=1;
    assert(size(A,2) == size(B,1), 'Inputs must be Multiplication compatiable.');
    [U1, Sigma1, V1] = randomized_partial_svd(A, comp);
    [U2, Sigma2, V2] = randomized_partial_svd(B, comp);
    
    a = min(1 * floor(log2(n)), n); 
   
    
    while (comp*a)<=n
        
        
        B_constructed=(U2.*Sigma2')*V2';
        delB=B-B_constructed;
        AB_constructed=((A*U2).*Sigma2')*V2';
        A_constructedDelB=(U1.*Sigma1')*(V1'*delB);
        A_constructed=(U1.*Sigma1')*(V1');
	    M=(AB_constructed+A_constructedDelB);
       
        err=norm(AB-M,'fro')/AB_fro;

        
        
        if err<tol
            
            return;
        end
        
        comp=comp+1;
        
        [U1, Sigma1, V1] = randomized_partial_svd(A, comp);
        [U2, Sigma2, V2] = randomized_partial_svd(B, comp);
        
      
    end

end
