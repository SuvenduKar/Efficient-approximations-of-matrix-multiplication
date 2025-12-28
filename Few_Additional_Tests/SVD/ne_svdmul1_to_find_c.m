% This is first order SVD based approximation of A*B, without considering
% the error computation time.
function [c1,c2,c3,c4,c5,err]=ne_svdmul1_to_find_c(A,B,comp)
    AB=A*B;
    AB_fro=norm(AB,'fro');
    A_fro=norm(A,'fro');
    B_fro=norm(B,'fro');
    
    [m,n]=size(A);
    
    assert(size(A,2) == size(B,1), 'Inputs must be Multiplication compatiable.');
    [U1, Sigma1, V1] = randomized_partial_svd(A, comp);
    [U2, Sigma2, V2] = randomized_partial_svd(B, comp);
    [~, indicesA] = sort(Sigma1, 'descend');
    [~, indicesB] = sort(Sigma2, 'descend');
    A_constructed = zeros(m,n);
    B_constructed = zeros(m,n);
    AB_constructed=zeros(m,n);
    A_constructed_delB=zeros(m,n);
    A_constructed_B_constructed=zeros(m,n);
    
    delA_B_constructed=zeros(m,n);
    
    

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
        delA_B_constructed=delA_B_constructed+Sigma2(k2)*(delA*U2(:,k2))*V2(:,k2)';
        A_constructed_B_constructed=A_constructed_B_constructed+Sigma2(k2)*(A_constructed*U2(:,k2))*V2(:,k2)';

        A_constructed_delB=A_constructed_delB+Sigma1(k1)*U1(:,k1)*(V1(:,k1)'*delB);
    end
    
    [err]=err_in_approximation(AB,AB_constructed,A_constructed_delB,AB_fro);
    c1=AB_fro/(A_fro*B_fro);
    c3=norm(A_constructed_delB,'fro')/(norm(A_constructed,'fro')*norm(delB,'fro'));
    c2=norm(A_constructed_B_constructed,'fro')/(norm(A_constructed,'fro')*norm(B_constructed,'fro'));
    c4=norm(delA_B_constructed,'fro')/(norm(delA,'fro')*norm(B_constructed,'fro'));
    c5=norm(AB-A_constructed_B_constructed-A_constructed_delB-delA_B_constructed,'fro')/(norm(delA,'fro')*norm(delB,'fro'));
    c1=1/c1;
    c2=1/c2;
    c3=1/c3;
    c4=1/c4;
    c5=1/c5;
    
            
        
        
       
    

end