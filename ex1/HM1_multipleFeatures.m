clear all
close all
clc

% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);
% Print out some data points
% First 10 examples from the dataset
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

% Scale features and set them to zero mean
[X, mu, sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];

% Run gradient descent
% Choose some alpha value
alpha = 0.1;
num_iters = 400;
% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
% Display gradient descent's result
fprintf('Theta computed from gradient descent:\n%f,\n%f',theta(1),theta(2))

% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
x0 = 1;
x1 = 1650;
x2 = 3;
xx = [x1, x2];
xx = (xx-mu)./sigma;
xx = [1 xx]';

price = theta'*xx; % Enter your price formula here
% ============================================================
fprintf('\n Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f', price);

figure()
plot(1:num_iters, J_history)
xlabel('# iteration')
ylabel('J(\theta)')

figure()
plot3(X(:,2),X(:,3),y,'rx')
hold on
plot3(X(:,2),X(:,3),X*theta, '-')

