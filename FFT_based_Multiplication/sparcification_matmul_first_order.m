function [comp,err]=sparcification_matmul_first_order(A,B,tol)
    [m1,n1]=size(A);
    [m2,n2]=size(B);
    AB=A*B;
    M=zeros(m1,n2);
    if n1~=m2
        fprintf('Multiplication is not possible due to dimension mismatch')
        return;
    end
    W=zeros(n1,n1);
    for i=1:n1
        for j=1:n1
            W(i,j)=exp(-1i*2*pi*(i-1)*(j-1)/n1);
        end
    end
    W=(1/sqrt(n1))*W;
    A_t=A*W';
    B_t=W*B;
    comp=1;
    r1=min(floor(comp*log2(n1)),n1);
    A_approx=top_k_abs_values(A_t, r1);
    r2=min(floor(comp*log2(n2)),n2);
    B_approx=top_k_abs_values(B_t, r2);
    M=A_t*B_approx+A_approx*(B_t-B_approx);
    err=norm(AB-M,'fro')/norm(AB,'fro');
    if err<tol
        return;
    end

    while err>=tol
        comp=comp+1;
        r1=min(floor(comp*log2(n1)),n1);
        A_approx=top_k_abs_values(A_t, r1);
        r2=min(floor(comp*log2(n2)),n2);
        B_approx=top_k_abs_values(B_t, r2);
        M=A_t*B_approx+A_approx*(B_t-B_approx);
        err=norm(AB-M,'fro')/norm(AB,'fro');
        if r1==n1 || r2==n2
            return;
        end
    end
end
    
    
    
        
