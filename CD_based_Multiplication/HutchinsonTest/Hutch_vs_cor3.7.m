
%% ==================== PARAMETERS ====================
repeat = 5;
order = 700;
tol_he = 0.05;
tol_cd=0.05;


%% ==================== MATRIX CASES ====================
cases = {
    'Toeplitz \& Toeplitz', @generate_toeplitz_mat_uniform, @generate_toeplitz_mat_uniform;
    'Symmetric \& Toeplitz', @(n) sym_rand(n), @generate_toeplitz_mat_uniform;
    'Toeplitz \& Hankel', @generate_toeplitz_mat_uniform, @generate_hankel_mat_uniform;
    'General \& Toeplitz', @(n) rand(n), @generate_toeplitz_mat_uniform;
    'Symmetric \& Symmetric', @(n) sym_rand(n), @(n) sym_rand(n);
    'Symmetric \& Hankel', @(n) sym_rand(n), @generate_hankel_mat_uniform;
    'General \& Symmetric', @(n) rand(n), @(n) sym_rand(n);
    'Block Toeplitz \& Block Toeplitz', @(n) block_toeplitz(n,10), @(n) block_toeplitz(n,10);
    'Images', @(n) image_norm_resize('Matrices/cat.12472.jpg',n), @(n) image_norm_resize('Matrices/dog.55.jpg',n);
    'Hankel \& Hankel', @generate_hankel_mat_uniform, @generate_hankel_mat_uniform;
    'General \& Hankel', @(n) rand(n), @generate_hankel_mat_uniform;
    'Kappa \& Toeplitz', @kappa_mat, @generate_toeplitz_mat_uniform;
    'Kappa \& Kappa', @kappa_mat, @kappa_mat;
    'Type-3 \& Toeplitz', @type3mat, @generate_toeplitz_mat_uniform;
    'Type-1 \& Toeplitz', @type1mat, @generate_toeplitz_mat_uniform;
    'Kappa \& General', @kappa_mat, @(n) rand(n);
    'General \& General', @(n) rand(n), @(n) rand(n);
    'Type-2 \& Type-2', @type2mat, @type2mat;
    'Type-1 \& Type-1', @type1mat, @type1mat;
    'Type-3 \& Type-3', @type3mat, @type3mat;
};

%% ==================== FUNCTION TO RUN ONE EXPERIMENT ====================
function stats = run_experiment(Afun,Bfun,repeat,order,tol_he,tol_cd)
    sample_ab=zeros(repeat,1);
    sample_dadb=zeros(repeat,1);
    sample_comps=zeros(repeat,1);
    frob_norm_ab_list=zeros(repeat,1);
    frob_norm_dadb_list=zeros(repeat,1);

    rhs_3point7_percentage=zeros(repeat,1);
    lhs_3point7=zeros(repeat,1);
    he_lhs_3point7=zeros(repeat,1);
    maxk_ab=0;
    maxk_dadb=0;

    for i=1:repeat
        A=Afun(order); B=Bfun(order);
        AB=A*B;
        frob_norm_ab=norm(AB,'fro'); frob_norm_ab_list(i)=frob_norm_ab;

        [comp,~,A_constructed,B_constructed,M]=cdmul1(A,B,tol_cd);
        sample_comps(i)=comp;
        delA=A-A_constructed; delB=B-B_constructed;
        dadb=delA*delB;
        frob_norm_dadb=norm(dadb,'fro'); frob_norm_dadb_list(i)=frob_norm_dadb;
        lhs_3point7(i)=frob_norm_dadb/frob_norm_ab;
        
        rhs_3point7_percentage(i)=(norm(delA,'fro')*norm(delB,'fro'))/(sqrt(size(A,2))*norm(M,'fro'));

        samples=(1:ceil(log2(size(A,2))):size(A,2));
        flag_ab=0; flag_dadb=0;
        for j=1:length(samples)
            k=samples(j);
            
            G = 2*(randi([0,1], size(A,2), k)) - 1;
            if flag_ab==0, est_ab=norm(AB*G,'fro')/sqrt(k); end
            if abs(est_ab-frob_norm_ab)/frob_norm_ab<tol_he && flag_ab==0
                sample_ab(i)=k; maxk_ab=max(maxk_ab,k); flag_ab=1;
            end
            if flag_dadb==0, est_dadb=norm(dadb*G,'fro')/sqrt(k); end
            if abs(est_dadb-frob_norm_dadb)/frob_norm_dadb<tol_he && flag_dadb==0
                sample_dadb(i)=k; maxk_dadb=max(maxk_dadb,k); flag_dadb=1;
            end
            if flag_ab*flag_dadb==1, he_lhs_3point7(i)=est_dadb/est_ab; break; end
        end
    end

    stats.frobAB = mean(frob_norm_ab_list);
    stats.frobDADB = mean(frob_norm_dadb_list);
    stats.maxkAB = maxk_ab; stats.avgkAB = mean(sample_ab);
    stats.maxkDADB = maxk_dadb; stats.avgkDADB = mean(sample_dadb);
    stats.rhs = 100*mean(rhs_3point7_percentage);
    stats.rmse = 100*sqrt(mean((lhs_3point7 - rhs_3point7_percentage).^2));
    stats.comps = floor(mean(sample_comps));
    stats.rmse_he=100*sqrt(mean((lhs_3point7-he_lhs_3point7).^2));

end

%% ==================== FUNCTION FOR LLM CASES ====================
function stats = run_llm_case(Acell,Bcell,repeat,order,tol_he,tol_cd)
    ncase = length(Acell);
    frobAB=zeros(ncase,1); frobDADB=zeros(ncase,1);
    maxkAB=zeros(ncase,1); avgkAB=zeros(ncase,1);
    maxkDADB=zeros(ncase,1); avgkDADB=zeros(ncase,1);
    rhs=zeros(ncase,1); comps=zeros(ncase,1);
    rmse=zeros(ncase,1);rmse_he=zeros(ncase,1);

    for i=1:ncase
        Afun = @(n) Acell{i}; Bfun = @(n) Bcell{i};
        s = run_experiment(Afun,Bfun,repeat,order,tol_he,tol_cd);
        frobAB(i)=s.frobAB; frobDADB(i)=s.frobDADB;
        maxkAB(i)=s.maxkAB; avgkAB(i)=s.avgkAB;
        maxkDADB(i)=s.maxkDADB; avgkDADB(i)=s.avgkDADB;
        rhs(i)=s.rhs; rmse(i)=s.rmse;comps(i)=s.comps;rmse_he(i)=s.rmse_he;
    end

    % Aggregate with maximum across all LLM products
    stats.frobAB = max(frobAB); stats.frobDADB = max(frobDADB);
    stats.maxkAB = max(maxkAB); stats.avgkAB = max(avgkAB);
    stats.maxkDADB = max(maxkDADB); stats.avgkDADB = max(avgkDADB);
    stats.rhs = max(rhs); stats.rmse=max(rmse);stats.comps = max(comps);stats.rmse_he=max(rmse_he);
end

% %% ==================== RUN ALL NORMAL CASES ====================
for i=1:size(cases,1)
    name = cases{i,1};
    Afun = cases{i,2}; Bfun = cases{i,3};
    stats = run_experiment(Afun,Bfun,repeat,order,tol_he,tol_cd);
    fprintf(['%s & \\makecell{%.2e \\\\ %.2e} & \\makecell{%d \\\\ %d} & ' ...
        '\\makecell{%d \\\\ %d} & %.2e & \\makecell{%.2e \\\\ %.2e} & %d \\\\\n\\hline\n'],...
        name, stats.frobAB, stats.frobDADB, floor(stats.maxkAB), floor(stats.avgkAB), ...
        floor(stats.maxkDADB), floor(stats.avgkDADB), ...
        stats.rhs, stats.rmse_he,stats.rmse, floor(stats.comps));
end

% ==================== RUN LLM-1 ====================
Q1 = full(load('Matrices/Q1.mat').A);
K1 = full(load('Matrices/K1.mat').A);
V1 = full(load('Matrices/V1.mat').A);
I1 = full(load('Matrices/I1.mat').A);
Acell = {Q1, K1, V1, Q1};
Bcell = {I1, I1, I1, K1'};
stats = run_llm_case(Acell,Bcell,1,order,tol_he, tol_cd);
fprintf(['LLM-1($Q_1I_1,K_1I_1,V_1I_1,Q_1K_1^T$) & \\makecell{%.2e \\\\ %.2e} & \\makecell{%d \\\\ %d} & ' ...
        '\\makecell{%d \\\\ %d} & %.2e & \\makecell{%.2e \\\\ %.2e} & %d \\\\\n\\hline\n'],...
        stats.frobAB, stats.frobDADB, floor(stats.maxkAB), floor(stats.avgkAB), ...
        floor(stats.maxkDADB), floor(stats.avgkDADB), ...
        stats.rhs, stats.rmse_he,stats.rmse, floor(stats.comps));


%% ==================== RUN LLM-2 ====================
Q2 = full(load('Matrices/Q2.mat').A);
K2 = full(load('Matrices/K2.mat').A);
V2 = full(load('Matrices/V2.mat').A);
I2 = full(load('Matrices/I2.mat').A);
Acell = {Q2, K2, V2, Q2};
Bcell = {I2, I2, I2, K2'};
stats = run_llm_case(Acell,Bcell,repeat,order,tol_he,tol_cd);
fprintf(['LLM-2($Q_2I_2,K_2I_2,V_2I_2,Q_2K_2^T$) & \\makecell{%.2e \\\\ %.2e} & \\makecell{%d \\\\ %d} & ' ...
        '\\makecell{%d \\\\ %d} & %.2e & \\makecell{%.2e \\\\ %.2e} & %d \\\\\n\\hline\n'],...
        stats.frobAB, stats.frobDADB, floor(stats.maxkAB), floor(stats.avgkAB), ...
        floor(stats.maxkDADB), floor(stats.avgkDADB), ...
        stats.rhs, stats.rmse_he,stats.rmse, floor(stats.comps));


%% ==================== RUN BUS MATRICES ====================
Bus494 = full(load('Matrices/494_bus.mat').Problem.A);
Acell = {Bus494}; Bcell = {Bus494};
stats = run_llm_case(Acell,Bcell,repeat,order,tol_he,tol_cd);

fprintf(['Bus494 & \\makecell{%.2e \\\\ %.2e} & \\makecell{%d \\\\ %d} & ' ...
        '\\makecell{%d \\\\ %d} & %.2e & \\makecell{%.2e \\\\ %.2e} & %d \\\\\n\\hline\n'],...
        stats.frobAB, stats.frobDADB, floor(stats.maxkAB), floor(stats.avgkAB), ...
        floor(stats.maxkDADB), floor(stats.avgkDADB), ...
        stats.rhs, stats.rmse_he,stats.rmse, floor(stats.comps));


Bus662 = full(load('Matrices/662_bus.mat').Problem.A);
Acell = {Bus662}; Bcell = {Bus662};
stats = run_llm_case(Acell,Bcell,repeat,order,tol_he,tol_cd);
fprintf(['Bus662 & \\makecell{%.2e \\\\ %.2e} & \\makecell{%d \\\\ %d} & ' ...
        '\\makecell{%d \\\\ %d} & %.2e & \\makecell{%.2e \\\\ %.2e} & %d \\\\\n\\hline\n'],...
        stats.frobAB, stats.frobDADB, floor(stats.maxkAB), floor(stats.avgkAB), ...
        floor(stats.maxkDADB), floor(stats.avgkDADB), ...
        stats.rhs, stats.rmse_he,stats.rmse, floor(stats.comps));






















% %% ==================== PARAMETERS ====================
% repeat = 50;
% order = 700;
% tol = 0.005;
% comp = 5;
% 
% %% ==================== MATRIX CASES ====================
% cases = {
%     'Toeplitz \& Toeplitz', @generate_toeplitz_mat_uniform, @generate_toeplitz_mat_uniform;
%     'Symmetric \& Toeplitz', @(n) sym_rand(n), @generate_toeplitz_mat_uniform;
%     'Toeplitz \& Hankel', @generate_toeplitz_mat_uniform, @generate_hankel_mat_uniform;
%     'General \& Toeplitz', @(n) rand(n), @generate_toeplitz_mat_uniform;
%     'Symmetric \& Symmetric', @(n) sym_rand(n), @(n) sym_rand(n);
%     'Symmetric \& Hankel', @(n) sym_rand(n), @generate_hankel_mat_uniform;
%     'General \& Symmetric', @(n) rand(n), @(n) sym_rand(n);
%     'Block Toeplitz \& Block Toeplitz', @(n) block_toeplitz(n,10), @(n) block_toeplitz(n,10);
%     'Images', @(n) image_norm_resize('Matrices/cat.12472.jpg',n), @(n) image_norm_resize('Matrices/dog.55.jpg',n);
%     'Hankel \& Hankel', @generate_hankel_mat_uniform, @generate_hankel_mat_uniform;
%     'General \& Hankel', @(n) rand(n), @generate_hankel_mat_uniform;
%     'Kappa \& Toeplitz', @kappa_mat, @generate_toeplitz_mat_uniform;
%     'Kappa \& Kappa', @kappa_mat, @kappa_mat;
%     'Type-3 \& Toeplitz', @type3mat, @generate_toeplitz_mat_uniform;
%     'Type-1 \& Toeplitz', @type1mat, @generate_toeplitz_mat_uniform;
%     'Kappa \& General', @kappa_mat, @(n) rand(n);
%     'General \& General', @(n) rand(n), @(n) rand(n);
%     'Type-2 \& Type-2', @type2mat, @type2mat;
%     'Type-1 \& Type-1', @type1mat, @type1mat;
%     'Type-3 \& Type-3', @type3mat, @type3mat;
% };
% 
% %% ==================== FUNCTION TO RUN ONE EXPERIMENT ====================
% function stats = run_experiment(Afun,Bfun,repeat,order,tol,comp)
%     sample_ab=zeros(repeat,1);
%     sample_dadb=zeros(repeat,1);
%     frob_norm_ab_list=zeros(repeat,1);
%     frob_norm_dadb_list=zeros(repeat,1);
%     lhs_3point7=zeros(repeat,1);
%     rhs_3point7=zeros(repeat,1);
%     maxk_ab=0;
%     maxk_dadb=0;
% 
%     for i=1:repeat
%         A=Afun(order); B=Bfun(order);
%         AB=A*B;
%         frob_norm_ab=norm(AB,'fro'); frob_norm_ab_list(i)=frob_norm_ab;
% 
%         [~,A_constructed,B_constructed,M]=cdmul1(A,B,comp);
% 
%         delA=A-A_constructed; delB=B-B_constructed;
%         dadb=delA*delB;
%         frob_norm_dadb=norm(dadb,'fro'); frob_norm_dadb_list(i)=frob_norm_dadb;
% 
%         lhs_3point7(i)=frob_norm_dadb/frob_norm_ab;
%         rhs_3point7(i)=(norm(delA,'fro')*norm(delB,'fro'))/(sqrt(size(A,2))*norm(M,'fro'));
% 
%         samples=(1:ceil(log2(size(A,2))):size(A,2));
%         flag_ab=0; flag_dadb=0;
%         for j=1:length(samples)
%             k=samples(j);
%             %G=randn(size(A,2),k);
%             G = 2*(randi([0,1], size(A,2), k)) - 1;
%             if flag_ab==0, est_ab=norm(AB*G,'fro')/sqrt(k); end
%             if abs(est_ab-frob_norm_ab)/frob_norm_ab<tol && flag_ab==0
%                 sample_ab(i)=k; maxk_ab=max(maxk_ab,k); flag_ab=1;
%             end
%             if flag_dadb==0, est_dadb=norm(dadb*G,'fro')/sqrt(k); end
%             if abs(est_dadb-frob_norm_dadb)/frob_norm_dadb<tol && flag_dadb==0
%                 sample_dadb(i)=k; maxk_dadb=max(maxk_dadb,k); flag_dadb=1;
%             end
%             if flag_ab*flag_dadb==1, break; end
%         end
%     end
% 
%     stats.frobAB = mean(frob_norm_ab_list);
%     stats.frobDADB = mean(frob_norm_dadb_list);
%     stats.maxkAB = maxk_ab; stats.avgkAB = mean(sample_ab);
%     stats.maxkDADB = maxk_dadb; stats.avgkDADB = mean(sample_dadb);
%     stats.lhs = mean(lhs_3point7);
%     stats.rhs = mean(rhs_3point7);
%     %stats.varrhs = std(rhs_3point7);
%     stats.varrhs = sqrt( sum( (rhs_3point7 - mean(lhs_3point7)).^2 ) / repeat );
% end
% 
% %% ==================== FUNCTION FOR LLM CASES ====================
% function stats = run_llm_case(Acell,Bcell,repeat,order,tol,comp)
%     ncase = length(Acell);
%     frobAB=zeros(ncase,1); frobDADB=zeros(ncase,1);
%     maxkAB=zeros(ncase,1); avgkAB=zeros(ncase,1);
%     maxkDADB=zeros(ncase,1); avgkDADB=zeros(ncase,1);
%     lhs=zeros(ncase,1); rhs=zeros(ncase,1); varrhs=zeros(ncase,1);
% 
%     for i=1:ncase
%         Afun = @(n) Acell{i}; Bfun = @(n) Bcell{i};
%         s = run_experiment(Afun,Bfun,repeat,order,tol,comp);
%         frobAB(i)=s.frobAB; frobDADB(i)=s.frobDADB;
%         maxkAB(i)=s.maxkAB; avgkAB(i)=s.avgkAB;
%         maxkDADB(i)=s.maxkDADB; avgkDADB(i)=s.avgkDADB;
%         lhs(i)=s.lhs; rhs(i)=s.rhs; varrhs(i)=s.varrhs;
%     end
% 
%     % Aggregate with maximum across all LLM products
%     stats.frobAB = max(frobAB); stats.frobDADB = max(frobDADB);
%     stats.maxkAB = max(maxkAB); stats.avgkAB = max(avgkAB);
%     stats.maxkDADB = max(maxkDADB); stats.avgkDADB = max(avgkDADB);
%     stats.lhs = max(lhs); stats.rhs = max(rhs); stats.varrhs = max(varrhs);
% end
% 
% % %% ==================== RUN ALL NORMAL CASES ====================
% for i=1:size(cases,1)
%     name = cases{i,1};
%     Afun = cases{i,2}; Bfun = cases{i,3};
%     stats = run_experiment(Afun,Bfun,repeat,order,tol,comp);
%     fprintf(['%s & %.2e & \\makecell{%.2f \\\\ %.2f} & %.2e & ' ...
%         '\\makecell{%.2f \\\\ %.2f} & %.2e & %.2e & %.2e \\\\\n\\hline\n'],...
%         name, stats.frobAB, stats.maxkAB, stats.avgkAB, ...
%         stats.frobDADB, stats.maxkDADB, stats.avgkDADB, ...
%         stats.lhs, stats.rhs, stats.varrhs);
% end
% 
% % ==================== RUN LLM-1 ====================
% Q1 = full(load('Matrices/Q1.mat').A);
% K1 = full(load('Matrices/K1.mat').A);
% V1 = full(load('Matrices/V1.mat').A);
% I1 = full(load('Matrices/I1.mat').A);
% Acell = {Q1, K1, V1, Q1};
% Bcell = {I1, I1, I1, K1'};
% stats = run_llm_case(Acell,Bcell,5,order,tol,comp);
% fprintf(['LLM-1($Q_1I_1,K_1I_1,V_1I_1,Q_1K_1^T$) & %.2e & ' ...
%     '\\makecell{%.2f \\\\ %.2f} & %.2e & ' ...
%     '\\makecell{%.2f \\\\ %.2f} & %.2e & %.2e & %.2e \\\\\n\\hline\n'],...
%     stats.frobAB, stats.maxkAB, stats.avgkAB, stats.frobDADB,...
%     stats.maxkDADB, stats.avgkDADB, stats.lhs, stats.rhs, stats.varrhs);
% 
% %% ==================== RUN LLM-2 ====================
% Q2 = full(load('Matrices/Q2.mat').A);
% K2 = full(load('Matrices/K2.mat').A);
% V2 = full(load('Matrices/V2.mat').A);
% I2 = full(load('Matrices/I2.mat').A);
% Acell = {Q2, K2, V2, Q2};
% Bcell = {I2, I2, I2, K2'};
% stats = run_llm_case(Acell,Bcell,5,order,tol,comp);
% fprintf(['LLM-2($Q_2I_2,K_2I_2,V_2I_2,Q_2K_2^T$) & %.2e & ' ...
%     '\\makecell{%.2f \\\\ %.2f} & %.2e & ' ...
%     '\\makecell{%.2f \\\\ %.2f} & %.2e & %.2e & %.2e \\\\\n\\hline\n'],...
%     stats.frobAB, stats.maxkAB, stats.avgkAB, stats.frobDADB,...
%     stats.maxkDADB, stats.avgkDADB, stats.lhs, stats.rhs, stats.varrhs);
% 
% %% ==================== RUN BUS MATRICES ====================
% Bus494 = full(load('Matrices/494_bus.mat').Problem.A);
% Acell = {Bus494}; Bcell = {Bus494};
% stats = run_llm_case(Acell,Bcell,repeat,order,tol,comp);
% fprintf(['Bus494 \\& Bus494 & %.2e & \\makecell{%.2f \\\\ %.2f} & %.2e & ' ...
%     '\\makecell{%.2f \\\\ %.2f} & %.2e & %.2e & %.2e \\\\\n\\hline\n'],...
%     stats.frobAB, stats.maxkAB, stats.avgkAB, stats.frobDADB,...
%     stats.maxkDADB, stats.avgkDADB, stats.lhs, stats.rhs, stats.varrhs);
% 
% Bus662 = full(load('Matrices/662_bus.mat').Problem.A);
% Acell = {Bus662}; Bcell = {Bus662};
% stats = run_llm_case(Acell,Bcell,repeat,order,tol,comp);
% fprintf(['Bus662 \\& Bus662 & %.2e & \\makecell{%.2f \\\\ %.2f} & %.2e & ' ...
%     '\\makecell{%.2f \\\\ %.2f} & %.2e & %.2e & %.2e \\\\\n\\hline\n'],...
%     stats.frobAB, stats.maxkAB, stats.avgkAB, stats.frobDADB,...
%     stats.maxkDADB, stats.avgkDADB, stats.lhs, stats.rhs, stats.varrhs);