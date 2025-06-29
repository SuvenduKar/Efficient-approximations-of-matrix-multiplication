% Replace A,B with the matrices of your test-case

order=700;


tollist=[0.05;0.01];% Tolerance list

repeat=150; % Number of times to be repeated



for k=1:length(tollist)
    comp_list0=zeros(repeat,1);
    comp_list1=zeros(repeat,1);
    tol=tollist(k);
    for i=1:repeat
        A=generate_toeplitz_mat_uniform(order);

        B=generate_toeplitz_mat_uniform(order);
        [comp,err]=cdmul0(A,B,tol);
        comp_list0(i)=comp;
        fprintf(' Error in zeroth order multiplication is %f , with number of components %f \n',err,comp)
        [comp,err]=cdmul1(A,B,tol);
        comp_list1(i)=comp;
        fprintf(' Error in First order multiplication is %f , with number of components %f \n',err,comp)
    end
    fprintf('********************************Toeplitz-Toeplitz************************************\n')
    fprintf(' Zeroth order multiplication achieved tol = %f with number of components %f \n',tol,mean(comp_list0))

    fprintf(' First order multiplication achieved tol = %f with number of components %f \n',tol,mean(comp_list1))
    fprintf('********************************************************************\n')
end


