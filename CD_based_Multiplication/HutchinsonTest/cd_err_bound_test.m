% This is to plot the error bounds 


repeat=10;
eps=0.01;
n_list=[100;200;300;400;500;700];

cd0_err=zeros(length(n_list),1);
cd1_err=zeros(length(n_list),1);
thm36_err=zeros(length(n_list),1);
thm37_err=zeros(length(n_list),1);
for n_ind=1:length(n_list)
    n = n_list(n_ind);
    
    cd0_err_trial=zeros(repeat,1);
    cd1_err_trial=zeros(repeat,1);
    thm36_err_trial=zeros(repeat,1);
    thm37_err_trial=zeros(repeat,1);
    for i=1:repeat
        A=rand(n,n);%generate_toeplitz_mat_uniform(n);
        B=rand(n,n);
        
        [cd0_err_trial(i)]=cdmul0(A,B) ;
        [cd1_err_trial(i),thm36_err_trial(i),thm37_err_trial(i)]=cdmul1(A,B,eps);
    end
    
    cd0_err(n_ind)=mean(cd0_err_trial);
    cd1_err(n_ind)=mean(cd1_err_trial)+1e-6;
    thm36_err(n_ind)=mean(thm36_err_trial)+1e-6;
    thm37_err(n_ind)=mean(thm37_err_trial)+1e-6;

    
end

figure;


hold on; box on;


plot(n_list, cd0_err, '-d', ...   % square
    'LineWidth', 2.5, 'MarkerSize', 11);

plot(n_list, cd1_err, '-*', ...      % rhombus (diamond)
    'LineWidth', 2.5, 'MarkerSize', 8);



plot(n_list, thm36_err, '--s', ...     % star
    'LineWidth', 2.5, 'MarkerSize', 20);

plot(n_list, thm37_err, '--^', ...     % triangle
    'LineWidth', 2.5, 'MarkerSize', 16);

% Axes labels
xlabel('$n$', 'Interpreter','latex', 'FontSize', 22);
ylabel('Error', 'Interpreter','latex', 'FontSize', 22);

% Log scale on Y-axis
set(gca, 'YScale', 'log');
ylim([10^-3 10^0]); %If the plot is still not log-scaled due to narrow range, use this

set(gcf, 'PaperPositionMode','auto');
% Legend
lgd = legend({
              'CD-Zeroth Order', ...
              'CD-First Order', ...
              'Theorem 3.6', ...
              'Corollary 3.7'}, ...
             'Interpreter','latex', ...
             'FontSize', 24);

set(lgd, ...
    'Units','normalized', ...
    'Position',[0.50 0.55 0.35 0.35], ... % [x y width height]
    'Box','off');


% Axis formatting
set(gca, 'FontSize', 22, 'LineWidth', 1.2);
grid on;

hold off;



% Save figure using print
set(gcf, 'PaperUnits','inches');
set(gcf, 'PaperPosition',[0 0 5 3.8]);


exportgraphics(gcf, 'cd_err_bound_gg.png', 'Resolution', 600);
