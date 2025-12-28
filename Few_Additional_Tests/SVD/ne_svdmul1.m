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
    [~, indicesA] = sort(Sigma1, 'descend');
    [~, indicesB] = sort(Sigma2, 'descend');
    A_constructed = zeros(m,n);
    B_constructed = zeros(m,n);
    AB_constructed=zeros(m,n);
    AdelB_constructed=zeros(m,n);


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
        delB=B-B_constructed;
        for k=1:length(Sigma1)
            k1=indicesA(k);
            k2=indicesB(k);

            AB_constructed=AB_constructed+Sigma2(k2)*(A*U2(:,k2))*V2(:,k2)';

            AdelB_constructed=AdelB_constructed+Sigma1(k1)*U1(:,k1)*(V1(:,k1)'*delB);
        end
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
        [~, indicesA] = sort(Sigma1, 'descend');
        [~, indicesB] = sort(Sigma2, 'descend');
        A_constructed = zeros(m,n);
        B_constructed = zeros(m,n);
        AB_constructed=zeros(m,n);
        AdelB_constructed=zeros(m,n);
    end

end

