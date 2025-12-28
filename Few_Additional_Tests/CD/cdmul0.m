
function [err]=cdmul0(A,B)
    AB=A*B;
    [m,n]=size(A);
    assert(size(A,2) == size(B,1), 'Inputs must be Multiplication compatiable.');
    [RA]=cd_components(A); % Getting R_k s for A
    [RB]=cd_components(B); % Getting R_k s for B
    [indicesA] = top_indices(RA);
    [indicesB] = top_indices(RB);
    A_constructed = zeros(m,n);
    B_constructed = zeros(m,n);

    comp=5;
    a = min(5 * floor(log2(n)), n); 
    start=1;
    terminal=start+a;

    
        
    for k=start:terminal
        k1=indicesA(k);
        k2=indicesB(k);

        RAk=RA(:,:,k1);
        RBk=RB(:,:,k2);

        
        RAk = circulant_matrix(RAk);
        Dk1=diag(exp(1i * 2*pi * (k1-1) * (0:n-1) / n).'); 
        A_constructed=A_constructed+RAk*Dk1;

        
        RBk = circulant_matrix(RBk);
        Dk2=diag(exp(1i * 2*pi * (k2-1) * (0:n-1) / n).'); 
        B_constructed=B_constructed+RBk*Dk2;
    end
    err=norm(AB-A_constructed*B_constructed,'fro')/norm(AB,'fro'); % This multiplication can be made optimal in cost of computation following the corresponding algorithm in the paper
    

end
