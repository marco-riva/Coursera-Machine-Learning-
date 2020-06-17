clear all
close all
clc


data = load('ex1data2.txt'); % read comma separated data
X = data(:, 2); y = data(:, 3);

plotData(X,y)

m = length(X) % number of training examples
X = [ones(m, 1), data(:,2)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters
iterations = 2000;
alpha = 0.01;

% Run gradient descent:
% Compute theta
[theta,J_history] = gradientDescent(X, y, theta, alpha, iterations);
% Print theta to screen
% Display gradient descent's result
fprintf('Theta computed from gradient descent:\n%f,\n%f',theta(1),theta(2))
% Plot the linear fit
hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

figure()
    plot(1:iterations, J_history)
    xlabel('# iteration')
    ylabel('J(\theta)')
  