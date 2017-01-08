% Implement the gradient for the cost function of regularized linear regression
% Your function will take in:
%
% theta∈ℝn+1: A (n+1)-dimensional vector representing parameters of the regularized l% inear regression model, including a term for the bias
% X∈ℝm×n+1: A m×n+1 matrix containing the features for each data point.
% y∈ℝm: A m-dimensional vector representing the y-values for each example.
% lambda∈ℝ: The regularization parameter.
%
% ∂J(θ)∂θ0=1m ∑i=1m(hθ(x(i))−y(i))x(i)0
% ∂J(θ)∂θj=(1m ∑i=1m(hθ(x(i))−y(i))x(i)j)+λmθj          j∈{1,2...n}
%
% Note that in Octave, vectors are indexed from 1 onwards; that means that theta(1) in your code refers to theta(0)


function grad = grad(theta, X, y, lambda)

	h = sigmoid(X*theta);
	m = size(y)(1);

	%cost function
	J = (sum(-y .* log(h) - (1 - y) .* log(1 - h)))/m;

	%gradient
	grad = zeros(theta, 1);
	grad = ((X'*(h-y)))/m;
	grad = grad(:);


endfunction