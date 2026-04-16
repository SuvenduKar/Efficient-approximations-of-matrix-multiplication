
% Efficiency comparison for the randomized outer product vs SVD 1st Order.
% Change the A,B as per the test-case
orders=[100;500;1000;1500;2000;2500;3000;4000;5000];

tollist=[0.05;0.01]; % Tolerance list
repeat=50; % Number of times the test to be repeated



for k=1:length(tollist)
    randmul_time=zeros(length(orders),1);
    svd_time=zeros(length(orders),1);
    for o=1:length(orders)
        order=orders(o);
        time_list=zeros(repeat,1);
        comp_list1=zeros(repeat,1);
        time_list1=zeros(repeat,1);
        
        tol=tollist(k);

        for i=1:repeat
            A=type1mat(order);
    
            B=type1mat(order);
            [~,~,elapsedTime] = ne_low_rank_approx_algo(A, B, tol);
            time_list(i)=elpasedTime;
            [comp,err,time]=ne_svdmul1(A,B,tol);
            time_list1(i)=time;
            
            
            
        end
        %fprintf('********************************************************************\n')
        randmul_time(o) = mean(time_list);
        svd_time(o) = mean(time_list1);
        
        %fprintf('********************************************************************\n')
    end
    %plot graph randmul_time vs svd_time
    figure;
    plot(orders, randmul_time, '-o', orders, svd_time, '-x');
    xlabel('Order');
    ylabel('Time (seconds)');
    legend('Randomized Outer Product', 'SVD First Order');
    title('Comparison of Multiplication Times with tol=',tol);
    grid on;
    %save the picture with name as time_{tol}
    saveas(gcf, sprintf('randomized_outer_product_vs_svd_mul_time_%.2f.png', tol));
end




