% This is to plot the error bounds 

% Low rank approximation vs SVD 0th order vs SVD first order vs Theorem 3.6
% vs corollary 3.7
repeat=100;% Number of times the test to be repeated before taking the average 
eps=0.01;
n_list=[100;200;300;400;500;700];
lowrank_err=zeros(length(n_list),1);
svd0_err=zeros(length(n_list),1);
svd1_err=zeros(length(n_list),1);
thm36_err=zeros(length(n_list),1);
thm37_err=zeros(length(n_list),1);
for n_ind=1:length(n_list)
    n = n_list(n_ind);
    lowrank_err_trail=zeros(repeat,1);
    svd0_err_trial=zeros(repeat,1);
    svd1_err_trial=zeros(repeat,1);
    thm36_err_trial=zeros(repeat,1);
    thm37_err_trial=zeros(repeat,1);
    for i=1:repeat
        A=type2mat(n);
        B=type2mat(n);
        [lowrank_err_trail(i)]=ne_low_rank_approx_algo_fixed_comp(A, B);
        [svd0_err_trial(i)]=svdmul0_fixed_comp(A,B) ;
        [svd1_err_trial(i),thm36_err_trial(i),thm37_err_trial(i)]=svdmul1_fixed_comp(A,B,eps);
    end
    lowrank_err(n_ind)=mean(lowrank_err_trail);
    svd0_err(n_ind)=mean(svd0_err_trial);
    svd1_err(n_ind)=mean(svd1_err_trial);
    thm36_err(n_ind)=mean(thm36_err_trial);
    thm37_err(n_ind)=mean(thm37_err_trial);

    
end
figure;

hold on; box on;

plot(n_list, lowrank_err, '-s', ...   % square
    'LineWidth', 2.5, 'MarkerSize', 11);

plot(n_list, svd0_err, '-d', ...      % rhombus (diamond)
    'LineWidth', 2.5, 'MarkerSize', 11);

plot(n_list, svd1_err, '-o', ...      % circle
    'LineWidth', 2.5, 'MarkerSize', 15);

plot(n_list, thm36_err, '--*', ...     % star
    'LineWidth', 2.5, 'MarkerSize', 12);

plot(n_list, thm37_err, '--^', ...     % triangle
    'LineWidth', 2.5, 'MarkerSize', 11);

% Axes labels
xlabel('$n$', 'Interpreter','latex', 'FontSize', 22);
ylabel('Error', 'Interpreter','latex', 'FontSize', 22);

% Log scale on Y-axis
set(gca, 'YScale', 'log');
set(gcf, 'PaperPositionMode','auto');
% Legend
lgd = legend({'Low Rank Multiplication', ...
              'SVD-Zeroth Order', ...
              'SVD-First Order', ...
              'Theorem 3.6', ...
              'Corollary 3.7'}, ...
             'Interpreter','latex', ...
             'FontSize', 24);

set(lgd, ...
    'Units','normalized', ...
    'Position',[0.50 0.21 0.35 0.35], ... % [x y width height]
    'Box','off');


% Axis formatting
set(gca, 'FontSize', 22, 'LineWidth', 1.2);
grid on;

hold off;

% Save figure using print
set(gcf, 'PaperUnits','inches');
set(gcf, 'PaperPosition',[0 0 5 3.8]);


exportgraphics(gcf, 'svd_err_bound_t2t2.png', 'Resolution', 600);
