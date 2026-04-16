function [comp,err,A_constructed, B_constructed,M]=cdmul1(A,B,tol)
    AB=A*B;
    [m,n]=size(A);
    assert(size(A,2) == size(B,1), 'Inputs must be Multiplication compatiable.');
    [RA]=cd_components(A); % Getting R_k s for A
    [RB]=cd_components(B); % Getting R_k s for B
    [indicesA] = top_indices(RA);
    [indicesB] = top_indices(RB);
    A_constructed = zeros(m,n);
    B_constructed = zeros(m,n);

    comp=1;
    a = min(1 * floor(log2(n)), n); 
    start=1;
    terminal=start+a-1;

    while (comp*a)<=n % Check if all the components are exhausted
        
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
        M=(A*B_constructed)+(A_constructed*(B-B_constructed));
        err=norm(AB-M,'fro')/norm(AB,'fro'); % This multiplication can be made optimal in cost of computation following the corresponding algorithm in the paper
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
    end

end



















% function [err,A_constructed,B_constructed,M]=cdmul1(A,B,comp)
%     AB=A*B;
%     [m,n]=size(A);
%     assert(size(A,2) == size(B,1), 'Inputs must be Multiplication compatiable.');
%     [RA]=cd_components(A); % Getting R_k s for A
%     [RB]=cd_components(B); % Getting R_k s for B
%     [indicesA] = top_indices(RA);
%     [indicesB] = top_indices(RB);
%     A_constructed = zeros(m,n);
%     B_constructed = zeros(m,n);
% 
% 
%     a = min(comp * floor(log2(n)), n); 
%     start=1;
%     terminal=start+a;
% 
% 
% 
%     for k=start:terminal
%         k1=indicesA(k);
%         k2=indicesB(k);
% 
%         RAk=RA(:,:,k1);
%         RBk=RB(:,:,k2);
% 
% 
%         RAk = circulant_matrix(RAk);
%         Dk1=diag(exp(1i * 2*pi * (k1-1) * (0:n-1) / n).'); 
%         A_constructed=A_constructed+RAk*Dk1;
% 
% 
%         RBk = circulant_matrix(RBk);
%         Dk2=diag(exp(1i * 2*pi * (k2-1) * (0:n-1) / n).'); 
%         B_constructed=B_constructed+RBk*Dk2;
%     end
% 
%     M=(A*B_constructed)+(A_constructed*(B-B_constructed));
%     err=norm(AB-M,'fro')/norm(AB,'fro'); % This multiplication can be made optimal in cost of computation following the corresponding algorithm in the paper
% 
%     % mre1= (1/sqrt(1-eps))*(4*norm(delA,'fro')*norm(delB,'fro'))/(3*sqrt(n)*norm(A,'fro')*norm(B,'fro'));
%     % mre2=(norm(delA,'fro')*norm(delB,'fro'))/(sqrt(n)*sqrt(1-eps)*(norm(M,'fro')));
% end