function [Rk]=cd_components(A)
    
    [m,n]=size(A);

    omega=exp(1i*2*pi/n);
    % Get (orthogonal, normalized)  dft matrix 
    W=zeros(n,n);
    for i=1:n
        for j=1:n
            W(i,j)=omega^(-(i-1)*(j-1));
        end
    end
    W=(1/sqrt(n))*W;
    W_H=W'; %  This is conjugate transpose of W
    
    powers = omega .^ (0:n-1);               % row vector [omega^1, ..., omega^n]
    D = diag(powers); 
    
    % Get the C matrix
    C=zeros(n,n);
    C(1,n)=1;
    C(2:n,1:n-1)=eye(n-1);
    
    % Get A_tilde
    At=zeros(n,n);

    for j=0:n-1
        for i=0:n-1
            col=mod(i+j,n);
            At(i+1,j+1)=A(col+1,i+1); 
        end
    end
    
    
    Rk=zeros(n,1,n);
    B=W*At;
    for k=1:n
        
        Rk(:,:,k)=(1/sqrt(n))*(B(k,:).');% This is the first column of R_k in A= sum (R_kD^k)
       
        
    end
    
    %% Uncomment this phrase to verify if the reconstruction is valid or not 
    % result=zeros(n,n);
    % for i=1:n
    % 
    %     result=result+circulant_matrix(Rk(:,:,i))*D^(i-1);
    % end
    % fprintf(' The approximated A is\n');
    % disp(result);
    % disp(norm(A-result));
end 


