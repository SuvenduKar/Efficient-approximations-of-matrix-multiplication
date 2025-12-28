n_values=[50;100;500;1000;5000;10000];
comp_list=[1];
repeat=5;


for k=1:length(comp_list)
    comp=comp_list(k);
    c1_list=zeros(length(n_values),1);
    c2_list=zeros(length(n_values),1);
    c3_list=zeros(length(n_values),1);
    c4_list=zeros(length(n_values),1);
    c5_list=zeros(length(n_values),1);
    for j = 1:length(n_values)
        n = n_values(j);
        
        c1_trial = zeros(repeat,1);
        c2_trial = zeros(repeat,1);
        c3_trial = zeros(repeat,1);
        c4_trial = zeros(repeat,1);
        c5_trial = zeros(repeat,1);
        for r=1:repeat
            A=type1mat(n);
            %B=type1mat(n);
            B=A;
            [c1,c2,c3,c4,c5,~]=ne_svdmul1_to_find_c(A,B,comp);
            c1_trial(r) = c1;
            c2_trial(r) = c2;
            c3_trial(r) = c3;
            c4_trial(r) = c4;
            c5_trial(r) = c5;
        end
        c1_list(j) = mean(c1_trial);
        c2_list(j) = mean(c2_trial);
        c3_list(j) = mean(c3_trial);
        c4_list(j) = mean(c4_trial);
        c5_list(j) = mean(c5_trial);
    end
    % Plot
    figure;
    plot(n_values, c1_list, '-o', 'LineWidth', 1.5); hold on;
    plot(n_values, c2_list, '-s', 'LineWidth', 1.5);
    plot(n_values, c3_list, '-^', 'LineWidth', 1.5);
    plot(n_values, c4_list, '-d', 'LineWidth', 1.5);
    plot(n_values, c5_list, '-x', 'LineWidth', 1.5);
    hold off;
    
    xlabel('Matrix Dimension');
    ylabel('Value');
    legend('c1=||A||||B||/||AB||', 'c2=||t(A)||||t(B)||/||t(A)t(B)||', 'c3=||t(A)||||d(B)||/||t(A)d(B)||', 'c4=||d(A)||||t(B)||/||d(A)t(B)||', 'c5=||d(A)||||d(B)||/||d(A)d(B)||', 'Location', 'best');
    title(sprintf('c_i values vs n for comp=%d', comp));
    grid on;
    
    % ======== Y‑axis adjust ========
    C = [c1_list(:), c2_list(:), c3_list(:), c4_list(:), c5_list(:)];
    ymin = min(C, [], 'all');    
    ymax = max(C, [], 'all');    
    margin = 0.05 * (ymax - ymin);
    ylim([ymin - margin, ymax + margin]);
    % ===============================
    
    filename = sprintf('c_vs_n_comp_%d_new.png', comp);
    %exportgraphics(gcf, filename, 'Resolution', 300);

    print(gcf, filename, '-dpng', '-r300');
end
