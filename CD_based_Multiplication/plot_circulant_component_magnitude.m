% Example matrix (replace with your matrix)
% This is to see the pattern of magnitude of circulant components.
order=250;
A = rand(order,order);
[Rk]=cd_components(A);
% Compute singular values
s = zeros(1,order);
for i=1:order
    s(i)=norm(Rk(:,:,i),2);
end
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
ylim([-0.5 max(s)*1.05])

% Save the figure
exportgraphics(gca,'random_general_cd_magnitude.png','Resolution',1200);