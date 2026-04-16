% This is to plot singular-value decay pattern for various matrices.
% Example matrix (replace with your matrix)

order=250;
A = rand(order,order);

% Compute singular values

s=svd(A);
% Components index
components = 1:length(s);

% Create figure
%figure('Color',[1 1 1]);
figure('Color',[1 1 1],'Position',[100 100 500 360]);
% Plot singular values (thicker blue line)
plot(components, s, 'Color',[0 0.4470 0.7410], 'LineWidth',3);
hold on

% Labels
xlabel('Components','FontSize',18);
ylabel('Magnitude','FontSize',18);

% Axis formatting
set(gca,...
    'FontSize',18,...
    'LineWidth',1.5,...
    'Box','on',...
    'Color',[1 1 1]);
ax = gca;
% Shift x-axis slightly left so 0 appears inside the plot
xlim([-15 length(s)+10])
% Set x-axis tick locations
ax.XTick = 0:100:length(s)+10;
% Slight padding for y-axis
ylim([-2 max(s)*1.05])

% Save the figure
exportgraphics(gca,'random_general_svd_magnitude.png','Resolution',1200);