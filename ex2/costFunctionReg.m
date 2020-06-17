function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% Compute J(theta)
z = X*theta; % Argument of sigmoid function 
h = sigmoid(z); % Hypotesis
Cost = -y.*log(h) - (1-y).*log(1-h); % Cost
CostSum = sum(Cost); % sum over Cost

theta(1) = 0; % Set theta0 = 0 to avoid regularization  
ThetaSq = theta.^2;
SumThetaSq = sum(ThetaSq);

J = 1/m*CostSum + lambda/2/m*SumThetaSq; % J function

% Compute gradient J(theta): dJ(theta)/dtheta
Err = (h-y); % Error
ErrX = (Err.*X); % Error time X(i)
delta = 1/m*sum(ErrX) + lambda/m*theta'; % gradient (row vector)

grad = delta'; % Gradient 

% =============================================================

end
