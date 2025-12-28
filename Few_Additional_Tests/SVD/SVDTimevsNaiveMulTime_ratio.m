%TO FIND TIME RATIO
% Change the A,B as per the test-case
orders=[100;500;700;1000;5000;10000;15000];

tollist=[0.01]; % Tolerance list
repeat=2; % Number of times the test to be repeated



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
            % Type 1 - Type 1
            A=type1mat(order);


    
            B=type1mat(order);% Change the matrix type you want to test with
            
            
            [~,~,time]=ne_svdmul1(A,B,tol);
            t_start=tic;
            AB=A*B;
            elapsedTime=toc(t_start);
            %[~,~,elapsedTime] = ne_low_rank_approx_algo(A, B, tol);
            time_list(i)=elapsedTime/time;

            % Symmetric - Toeplitz 
            A=rand(order,order);
            A=A+A.';
            B=generate_toeplitz_mat_uniform(order);
           
            
            [~,~,time]=ne_svdmul1(A,B,tol);
            t_start=tic;
            AB=A*B;
            elapsedTime=toc(t_start);
            %[~,~,elapsedTime] = ne_low_rank_approx_algo(A, B, tol);
           
            time_list1(i)=elapsedTime/time;
            
            
            
        end
        %fprintf('********************************************************************\n')
        randmul_time(o) = mean(time_list);
        svd_time(o) = mean(time_list1);
        
        %fprintf('********************************************************************\n')
    end
    % %plot graph naivemul_time vs svd_time
    
    figure;
    
    % --- Smoother curves ---
    orders_fine = linspace(min(orders), max(orders), 300);
    randmul_fine = interp1(orders, randmul_time, orders_fine, 'pchip'); 
    svd_fine     = interp1(orders, svd_time, orders_fine, 'pchip');
    
    % --- Plot line + marker together ---

    h1 = plot(orders_fine, randmul_fine, 'b-', 'LineWidth', 4); 
    hold on
    h2 = plot(orders_fine, svd_fine, 'r-', 'LineWidth', 4);
    plot(orders, randmul_time, 'bo', 'MarkerSize', 16, 'LineWidth', 4);
    plot(orders, svd_time, 'r^', 'MarkerSize', 16, 'LineWidth', 4);

    % % IF REQUIRE SET BOUNDS ACCORINGLY --- Set the Y-axis ticks (1 to y_max+1) ---
    % y_max = ceil(max([randmul_time, svd_time])); % Take max from both datasets and round up
    % yticks(1:y_max+1); % Set Y ticks from 1 to y_max+1
    
    % --- Labels ---
    xlabel('Matrix Dimension', 'FontSize', 24);
    ylabel('Time-ratio', 'FontSize', 24);
    
    % --- Legend ---

    hLeg1 = plot(nan, nan, 'bo-', 'LineWidth', 4, 'MarkerSize', 16);
    hLeg2 = plot(nan, nan, 'r^-', 'LineWidth', 4, 'MarkerSize', 16);
    
    legend([hLeg1 hLeg2], ...
        {'Type-1 X Type-1', 'Symmetric X Toeplitz'}, ...
        'FontSize', 24, 'Location', 'northwest');
    
    grid on;
    ax = gca;
    ax.FontSize = 22;
    ax.LineWidth = 1.2;

  

    
    % --- Save as high-resolution PNG ---
    print(gcf, sprintf('SVD_vs_Naive_timeratio_tol_%.2f.png', tol), '-dpng', '-r600');

end















