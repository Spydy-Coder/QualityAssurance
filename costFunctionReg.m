function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
h=sigmoid(X*theta);
theta(1)=0;
J=(1/m)*(((-1*y)'*log(h))-((1-y)'*log(1-h)))+((lambda/(2*m))*sum(theta(2:end).^2));

grad_unreg=(1/m)*((X')*(h-y));
grad_reg=(lambda/m)*theta;
grad=grad_unreg+grad_reg;

end
