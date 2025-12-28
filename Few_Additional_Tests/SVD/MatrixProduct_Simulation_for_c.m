%% =========================================================================
% Simulation Code to Verify Constants for Matrix Product Bounds
% across Different Distributions
% =========================================================================
% This script computes the expected value of ||AB||_F / (||A||_F * ||B||_F) for
% random matrices with entries drawn from:
%   1. Uniform U(0,1)
%   2. Rademacher (±1)
%   3. Gaussian N(0,1)
%   4. Log-Normal
%   5. Student's t with ν=3 (heavy-tailed)
%
% Expected Results:
%   - Uniform: constant ≈ 0.75 (3/4)
%   - Signed Distribution: O(1/√n) scaling with constants
% =========================================================================

clear all; close all; clc;

%% =========================================================================
% PARAMETERS
% =========================================================================

% Matrix dimensions
n_values = [50, 100, 200, 300,500,700,1000,1500,3000,5000];  % Matrix size n (for n×n matrices)

% Number of trials per configuration
num_trials = 25;  % Increase for better statistical estimates (e.g., 100+)

% Random seed for reproducibility
rng(42);

% Display settings
fprintf('\n');
fprintf('============================================================================\n');
fprintf(' Matrix Product Bounds Across Distributions\n');
fprintf('============================================================================\n');
fprintf('Testing: E[||AB||_F / (||A||_F * ||B||_F)] for different matrix distributions\n');
fprintf('\n');

%% =========================================================================
% DATA STORAGE
% =========================================================================

% Store results for each distribution
results = struct();

% Initialize storage for each distribution
distributions_to_test = {'Uniform', 'Rademacher', 'Gaussian', 'LogNormal', 'StudentT'};

for i = 1:length(distributions_to_test)
    results.(distributions_to_test{i}) = struct('means', [], 'stds', [], 'scaled_constants', []);
end

%% =========================================================================
% 1. UNIFORM DISTRIBUTION U(0,1)
% =========================================================================
% Expected: c ≈ 0.75 (constant, independent of n)
% Theory: E[||AB||_F / (||A||_F * ||B||_F)] ≈ 3/4

fprintf('\n');
fprintf('1. UNIFORM DISTRIBUTION U(0,1)\n');
fprintf('---------------------------------------------------------------------------\n');
fprintf('Theory predicts: c = 3/4 = %.6f (constant, independent of n)\n', 3/4);
fprintf('---------------------------------------------------------------------------\n');
fprintf('   n    | E[ratio]  | Std Dev  | Deviation from 3/4\n');
fprintf('---------------------------------------------------------------------------\n');

uniform_means = [];
uniform_stds = [];

for n = n_values
    % Store individual ratios for each trial
    ratios = zeros(num_trials, 1);
    
    for trial = 1:num_trials
        % Generate random matrices with U(0,1) entries
        A = rand(n, n);
        B = rand(n, n);
        
        % Compute matrix product
        AB = A * B;
        
        % Compute Frobenius norms
        norm_AB = norm(AB, 'fro');
        norm_A = norm(A, 'fro');
        norm_B = norm(B, 'fro');
        
        % Compute the relative measure: ||AB||_F / (||A||_F * ||B||_F)
        ratios(trial) = norm_AB / (norm_A * norm_B);
    end
    
    % Compute statistics
    mean_ratio = mean(ratios);
    std_ratio = std(ratios);
    deviation = mean_ratio - 3/4;
    
    uniform_means = [uniform_means; mean_ratio];
    uniform_stds = [uniform_stds; std_ratio];
    
    fprintf('%5d  | %.6f | %.6f | %.6f\n', n, mean_ratio, std_ratio, deviation);
end

results.Uniform.means = uniform_means;
results.Uniform.stds = uniform_stds;

%% =========================================================================
% 2. RADEMACHER DISTRIBUTION (±1)
% =========================================================================
% Expected: E[ratio] ~ 1/√n, so c ≈ E[ratio] * √n ≈ 1.0-1.1

fprintf('\n');
fprintf('2. RADEMACHER DISTRIBUTION (±1)\n');
fprintf('---------------------------------------------------------------------------\n');
fprintf('Theory predicts: c/√n scaling, where c ≈ 1.0-1.1\n');
fprintf('---------------------------------------------------------------------------\n');
fprintf('   n    | E[ratio]  | Std Dev  | c (scaled √n) | c*√n from theory\n');
fprintf('---------------------------------------------------------------------------\n');

rademacher_means = [];
rademacher_stds = [];
rademacher_scaled = [];

for n = n_values
    ratios = zeros(num_trials, 1);
    
    for trial = 1:num_trials
        % Generate Rademacher matrices: entries are ±1 with equal probability
        A = 2 * randi([0, 1], n, n) - 1;  % Rademacher: {-1, +1}
        B = 2 * randi([0, 1], n, n) - 1;
        
        A = double(A);
        B = double(B);
        
        AB = A * B;
        
        norm_AB = norm(AB, 'fro');
        norm_A = norm(A, 'fro');
        norm_B = norm(B, 'fro');
        
        ratios(trial) = norm_AB / (norm_A * norm_B);
    end
    
    mean_ratio = mean(ratios);
    std_ratio = std(ratios);
    
    % Scale by √n to extract the constant c
    scaled_c = mean_ratio * sqrt(n);
    
    rademacher_means = [rademacher_means; mean_ratio];
    rademacher_stds = [rademacher_stds; std_ratio];
    rademacher_scaled = [rademacher_scaled; scaled_c];
    
    fprintf('%5d  | %.6f | %.6f | %.6f | ≈ 1 (theoretical)\n', n, mean_ratio, std_ratio, scaled_c);
end

results.Rademacher.means = rademacher_means;
results.Rademacher.stds = rademacher_stds;
results.Rademacher.scaled_constants = rademacher_scaled;

%% =========================================================================
% 3. GAUSSIAN DISTRIBUTION N(0,1)
% =========================================================================
% Expected: E[ratio] ~ 1/√n, c ≈ 0.9-1.0 

fprintf('\n');
fprintf('3. GAUSSIAN DISTRIBUTION N(0,1)\n');
fprintf('---------------------------------------------------------------------------\n');
fprintf('Theory predicts: c/√n scaling, where c ≈ 0.9-1.0\n');
fprintf('---------------------------------------------------------------------------\n');
fprintf('   n    | E[ratio]  | Std Dev  | c (scaled √n) | c*√n from theory\n');
fprintf('---------------------------------------------------------------------------\n');

gaussian_means = [];
gaussian_stds = [];
gaussian_scaled = [];

for n = n_values
    ratios = zeros(num_trials, 1);
    
    for trial = 1:num_trials
        % Generate Gaussian matrices: entries ~ N(0,1)
        A = randn(n, n);
        B = randn(n, n);
        
        AB = A * B;
        
        norm_AB = norm(AB, 'fro');
        norm_A = norm(A, 'fro');
        norm_B = norm(B, 'fro');
        
        ratios(trial) = norm_AB / (norm_A * norm_B);
    end
    
    mean_ratio = mean(ratios);
    std_ratio = std(ratios);
    scaled_c = mean_ratio * sqrt(n);
    
    gaussian_means = [gaussian_means; mean_ratio];
    gaussian_stds = [gaussian_stds; std_ratio];
    gaussian_scaled = [gaussian_scaled; scaled_c];
    
    fprintf('%5d  | %.6f | %.6f | %.6f | ≈ 1 (theoretical)\n', n, mean_ratio, std_ratio, scaled_c);
end

results.Gaussian.means = gaussian_means;
results.Gaussian.stds = gaussian_stds;
results.Gaussian.scaled_constants = gaussian_scaled;

%% =========================================================================
% 4. LOG-NORMAL DISTRIBUTION
% =========================================================================
% Expected: E[ratio] ~ 1/√n, but with c ≈ 1.2-1.5 (larger than Gaussian)
% Reason: Non-zero mean + heavy right tail

fprintf('\n');
fprintf('4. LOG-NORMAL DISTRIBUTION exp(N(0,1))\n');
fprintf('---------------------------------------------------------------------------\n');
fprintf('Theory predicts: c scaling, where c ≈ 1/e\n');

fprintf('---------------------------------------------------------------------------\n');
fprintf('   n    | E[ratio]  | Std Dev  | c  | c from theory\n');
fprintf('---------------------------------------------------------------------------\n');

lognormal_means = [];
lognormal_stds = [];
lognormal_scaled = [];

for n = n_values
    ratios = zeros(num_trials, 1);
    
    for trial = 1:num_trials
        % Generate log-normal matrices: entries ~ exp(N(0,1))
        A = lognrnd(0, 1, n, n);  % MATLAB function for log-normal
        B = lognrnd(0, 1, n, n);
        
        AB = A * B;
        
        norm_AB = norm(AB, 'fro');
        norm_A = norm(A, 'fro');
        norm_B = norm(B, 'fro');
        
        ratios(trial) = norm_AB / (norm_A * norm_B);
    end
    
    mean_ratio = mean(ratios);
    std_ratio = std(ratios);
    scaled_c = mean_ratio ;
    
    lognormal_means = [lognormal_means; mean_ratio];
    lognormal_stds = [lognormal_stds; std_ratio];
    lognormal_scaled = [lognormal_scaled; scaled_c];
    
    fprintf('%5d  | %.6f | %.6f | %.6f | ≈ 1/e (theoretical)\n', n, mean_ratio, std_ratio, scaled_c);
end

results.LogNormal.means = lognormal_means;
results.LogNormal.stds = lognormal_stds;
results.LogNormal.scaled_constants = lognormal_scaled;

%% =========================================================================
% 5. STUDENT'S t DISTRIBUTION (ν=3) - HEAVY-TAILED
% =========================================================================
% Expected: E[ratio] ~ 1/√n, but with c ≈ 1
% Reason: No finite 4th moment → extreme outliers

fprintf('\n');
fprintf('5. STUDENT''S t DISTRIBUTION (ν=3) - HEAVY-TAILED\n');
fprintf('---------------------------------------------------------------------------\n');
fprintf('Theory predicts: c/√n scaling, where c ≈ 1\n');
fprintf('WARNING: No finite 4th moment → highly variable results\n');
fprintf('---------------------------------------------------------------------------\n');
fprintf('   n    | E[ratio]  | Std Dev  | c (scaled √n) | c*√n from theory\n');
fprintf('---------------------------------------------------------------------------\n');

student_means = [];
student_stds = [];
student_scaled = [];

for n = n_values
    ratios = zeros(num_trials, 1);
    
    for trial = 1:num_trials
        % Generate Student's t matrices: entries ~ t(ν=3)
        % MATLAB: trnd(v, m, n) generates t-distributed random numbers
        A = trnd(3, n, n);
        B = trnd(3, n, n);
        
        AB = A * B;
        
        norm_AB = norm(AB, 'fro');
        norm_A = norm(A, 'fro');
        norm_B = norm(B, 'fro');
        
        ratios(trial) = norm_AB / (norm_A * norm_B);
    end
    
    mean_ratio = mean(ratios);
    std_ratio = std(ratios);
    scaled_c = mean_ratio * sqrt(n);
    
    student_means = [student_means; mean_ratio];
    student_stds = [student_stds; std_ratio];
    student_scaled = [student_scaled; scaled_c];
    
    fprintf('%5d  | %.6f | %.6f | %.6f | ≈ 1 (theoretical)\n', n, mean_ratio, std_ratio, scaled_c);
end

results.StudentT.means = student_means;
results.StudentT.stds = student_stds;
results.StudentT.scaled_constants = student_scaled;

%% =========================================================================
% SUMMARY TABLE
% =========================================================================

fprintf('\n');
fprintf('============================================================================\n');
fprintf('SUMMARY: ESTIMATED CONSTANTS c FOR ALL DISTRIBUTIONS\n');
fprintf('============================================================================\n');
fprintf('\n');
fprintf('Distribution    | Form        | c (at n=5000)  | Theory Prediction\n');
fprintf('----------------|-------------|---------------|-------------------\n');
fprintf('Uniform U(0,1)  | c           | %.6f        | 0.750000\n', uniform_means(end));
fprintf('Rademacher (±1) | c/√n        | %.6f        | 1\n', rademacher_scaled(end));
fprintf('Gaussian N(0,1) | c/√n        | %.6f        | 1\n', gaussian_scaled(end));
fprintf('Log-Normal      | c           | %.6f        | 1/e\n', lognormal_scaled(end));
fprintf('Student t(ν=3)  | c/√n        | %.6f        | 1\n', student_scaled(end));
fprintf('----------------|-------------|---------------|-------------------\n');

%% =========================================================================
% VISUALIZATION
% =========================================================================

% Create figure for comparison
figure('Name', 'Matrix Product Bounds across Distributions', 'NumberTitle', 'off');
set(gcf, 'Position', [100, 100, 1200, 800]);

% Plot 1: Uniform (constant)
subplot(2,3,1);
errorbar(n_values, uniform_means, uniform_stds, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
yline(3/4, '--r', 'LineWidth', 2, 'DisplayName', 'Theory (3/4)');
xlabel('Matrix Size (n)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('E[||AB||_F / (||A||_F * ||B||_F)]', 'FontSize', 11, 'FontWeight', 'bold');
title('1. Uniform U(0,1): Constant', 'FontSize', 12, 'FontWeight', 'bold');
grid on; legend('Empirical', 'Theory');
ylim([0.7, 0.8]);

% Plot 2: Rademacher (1/√n)
subplot(2,3,2);
errorbar(n_values, rademacher_means, rademacher_stds, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Matrix Size (n)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('E[||AB||_F / (||A||_F * ||B||_F)]', 'FontSize', 11, 'FontWeight', 'bold');
title('2. Rademacher (±1): O(1/√n)', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

% Plot 3: Gaussian (1/√n)
subplot(2,3,3);
errorbar(n_values, gaussian_means, gaussian_stds, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Matrix Size (n)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('E[||AB||_F / (||A||_F * ||B||_F)]', 'FontSize', 11, 'FontWeight', 'bold');
title('3. Gaussian N(0,1): O(1/√n)', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

% Plot 4: Log-Normal 
subplot(2,3,4);
errorbar(n_values, lognormal_means, lognormal_stds, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Matrix Size (n)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('E[||AB||_F / (||A||_F * ||B||_F)]', 'FontSize', 11, 'FontWeight', 'bold');
title('4. Log-Normal: Heavy Tail', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

% Plot 5: Student's t 
subplot(2,3,5);
errorbar(n_values, student_means, student_stds, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Matrix Size (n)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('E[||AB||_F / (||A||_F * ||B||_F)]', 'FontSize', 11, 'FontWeight', 'bold');
title('5. Student t(ν=3): O(1/√n) + Extreme Tails', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

% Plot 6: Scaled constants comparison
subplot(2,3,6);
hold on;
% For uniform (constant), scale differently for visualization
plot(n_values, 0.75*ones(size(n_values)), 'o-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Uniform');
plot(n_values, rademacher_scaled, 's-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Rademacher');
plot(n_values, gaussian_scaled, '^-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Gaussian');
plot(n_values, lognormal_scaled, 'd-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Log-Normal');
plot(n_values, student_scaled, 'v-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Student t(ν=3)');
xlabel('Matrix Size (n)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Constant c (scaled by √n for others)', 'FontSize', 11, 'FontWeight', 'bold');
title('Scaled Constants Comparison', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best');
grid on;
ylim([0.5, 2.5]);

sgtitle('Theorem 3.5 Extension: Matrix Product Bounds Across Distributions', ...
    'FontSize', 14, 'FontWeight', 'bold');

%% =========================================================================
% SAVE RESULTS
% =========================================================================

% Save to workspace variable for further analysis
save('matrix_product_simulation_results.mat', 'results', 'n_values', 'num_trials');

fprintf('\n');
fprintf('============================================================================\n');
fprintf('Simulation completed successfully!\n');
fprintf('Results saved to: matrix_product_simulation_results.mat\n');
fprintf('============================================================================\n');

%% =========================================================================
% END OF SCRIPT
% =========================================================================
