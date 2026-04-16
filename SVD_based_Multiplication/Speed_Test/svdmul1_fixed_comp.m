% This is first order SVD based approximation of A*B
function [err,mre1,mre2]=svdmul1_fixed_comp(A,B,eps)
    AB=A*B;
    AB_fro=norm(AB,'fro');
    
    [m,n]=size(A);
    comp=5;
    assert(size(A,2) == size(B,1), 'Inputs must be Multiplication compatiable.');
    [U1, Sigma1, V1] = randomized_partial_svd(A, comp);
    [U2, Sigma2, V2] = randomized_partial_svd(B, comp);
    [~, indicesA] = sort(Sigma1, 'descend');
    [~, indicesB] = sort(Sigma2, 'descend');
    A_constructed = zeros(m,n);
    B_constructed = zeros(m,n);
    AB_constructed=zeros(m,n);
    AdelB_constructed=zeros(m,n);


    a = min(1 * floor(log2(n)), n); 
    

    for k=1:length(Sigma1)
        k1=indicesA(k);
        k2=indicesB(k);

        A_constructed=A_constructed+Sigma1(k1)*U1(:,k1)*V1(:,k1)';

        B_constructed=B_constructed+Sigma2(k2)*U2(:,k2)*V2(:,k2)';
    end
    delB=B-B_constructed;
    delA=A-A_constructed;
    for k=1:length(Sigma1)
        k1=indicesA(k);
        k2=indicesB(k);

        AB_constructed=AB_constructed+Sigma2(k2)*(A*U2(:,k2))*V2(:,k2)';

        AdelB_constructed=AdelB_constructed+Sigma1(k1)*U1(:,k1)*(V1(:,k1)'*delB);
    end
    M=AB_constructed+AdelB_constructed;
    [err]=err_in_approximation(AB,AB_constructed,AdelB_constructed,AB_fro);
    mre1= (1/sqrt(1-eps))*(4*norm(delA,'fro')*norm(delB,'fro'))/(3*sqrt(n)*norm(A,'fro')*norm(B,'fro'));
    mre2=(norm(delA,'fro')*norm(delB,'fro'))/(sqrt(1-eps)*sqrt(n)*norm(M,'fro'));
        
        
end

