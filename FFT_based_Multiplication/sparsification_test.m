% change the order and the A,B as per the test-case
order=700;


tollist=[0.05;0.01]; % Tolerance list

repeat=100; % Number of times the test to be repeated 



for k=1:length(tollist)
    comp_list0=zeros(repeat,1);
    comp_list1=zeros(repeat,1);
    tol=tollist(k);
    for i=1:repeat
        A=kappa_mat(order);%rand(order,order);
        %mat=rand(order,order);
        B=rand(order,order);%generate_hankel_mat_uniform(order);

        [comp,err]=sparcification_matmul_zeroth_order(A,B,tol);
        comp_list0(i)=comp;
        %fprintf(' Error in zeroth order multiplication is %f , with number of components %f \n',err,comp)
        [comp,err]=sparcification_matmul_first_order(A,B,tol);
        comp_list1(i)=comp;
        %fprintf(' Error in First order multiplication is %f , with number of components %f \n',err,comp)
    end
    fprintf('********************************Symmetric -Toeplitz************************************\n')
    fprintf(' Zeroth order multiplication achieved tol = %f with number of components %f \n',tol,mean(comp_list0))

    fprintf(' First order multiplication achieved tol = %f with number of components %f \n',tol,mean(comp_list1))
    fprintf('********************************************************************\n')
end

