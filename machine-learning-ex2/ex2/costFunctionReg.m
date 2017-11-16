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



m = length(y);


z= X*theta;
hOfx = sigmoid(z)   %  1./(1+ exp(-1.*z));
a = -y.*log(hOfx);
b= (1-y).*log(1-hOfx);
c = (lambda/(2*m))* sum(theta(1:end, 2:end)(:))
J= ((1/m) *sum(a-b))+c;

J = 1 / m * sum(-y .* log(hOfx) - (1 - y) .* log(1 - hOfx)) + lambda / (2 * m) * sum(theta(2:end) .^ 2);

grad(1) =  sum((hOfx - y) .* X(:, 1))*(1 / m);


for i = 2:size(theta, 1)
    grad(i) = 1 / m * sum((hOfx - y) .* X(:, i)) + lambda / m * theta(i);
end




% =============================================================

end
