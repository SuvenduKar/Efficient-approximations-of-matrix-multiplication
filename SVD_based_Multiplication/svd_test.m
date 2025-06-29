

% Change the A,B as per the test-case
order=700;

tollist=[0.05;0.01]; % Tolerance list

repeat=150; % Number of times the test to be repeated



for k=1:length(tollist)
    comp_list=zeros(repeat,1);
    comp_list0=zeros(repeat,1);
    comp_list1=zeros(repeat,1);
    tol=tollist(k);
    for i=1:repeat
        A=generate_toeplitz_mat_uniform(order);

        B=generate_toeplitz_mat_uniform(order);
        [comp,err]=low_rank_approx_algo(A,B,tol);
        comp_list(i)=comp;
        fprintf(' Error in Low-rank approximation Algorithm is %f , with number of components %f \n',err,comp)
        [comp,err]=svdmul0(A,B,tol);
        comp_list0(i)=comp;
        fprintf(' Error in zeroth order multiplication is %f , with number of components %f \n',err,comp)
        [comp,err]=svdmul1(A,B,tol);
        comp_list1(i)=comp;
        fprintf(' Error in First order multiplication is %f , with number of components %f \n',err,comp)
    end
    fprintf('********************************Toeplitz-Toeplitz************************************\n')
    fprintf(' Low rank approximation Algorithm achieved tol = %f with number of components %f \n',tol,mean(comp_list))
    fprintf(' Zeroth order multiplication achieved tol = %f with number of components %f \n',tol,mean(comp_list0))

    fprintf(' First order multiplication achieved tol = %f with number of components %f \n',tol,mean(comp_list1))
    fprintf('********************************************************************\n')
end




