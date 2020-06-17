function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Compute J(theta)
h = X*theta; % hypotesis
SqErr = (h-y).^2;
J = 1/2/m*sum(SqErr);

theta(1) = 0; % Set theta0 = 0 to avoid regularization  
ThetaSq = theta.^2;
SumThetaSq = sum(ThetaSq);

J = 1/2/m*sum(SqErr) + lambda/2/m*SumThetaSq; % J function

% Compute gradient J(theta): dJ(theta)/dtheta
Err = (h-y); % Error
ErrX = (Err.*X); % Error time X(i)
delta = 1/m*sum(ErrX) + lambda/m*theta'; % gradient (row vector)

grad = delta'; % Gradient 

end
