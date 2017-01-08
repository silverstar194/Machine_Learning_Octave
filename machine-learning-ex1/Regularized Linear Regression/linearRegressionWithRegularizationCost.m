% Implement the cost function for regularized linear regression
%
%Your function will take in:
%
% theta∈ℝn+1: A (n+1)-dimensional vector representing parameters of the regularized l% inear regression model, including a term for the bias
% X∈ℝm×n+1: A m×n+1 matrix containing the features for each data point.
% y∈ℝm: A m-dimensional vector representing the y-values for each example.
% lambda∈ℝ: The regularization parameter.
%
%J(θ)=12m [∑mi=1(hθ(x(i))−y(i))2+λ ∑nj=1θ2j]
%
% Note that in Octave, vectors are indexed from 1 onwards; that means that theta(1) in your code refers to theta(0)




function cost = cost(theta, X, y, lambda)

	m = size(y)(1);
	h = X*theta;

	costTerm = (sum((h-y) .^ 2);

	regularizationTerm = ((sum(theta .^ 2) - theta(1)^2)*lambda);

	cost = (costTerm+regularizationTerm)/2m;

endfunction