function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
%Return values
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

%Cost function
J= 0.5 * sum(sum(    (  R .*(X*Theta')-Y  ).^2))+ (lambda / 2) * sum(sum(Theta .^ 2)) + (lambda / 2) * sum(sum(X .^ 2));


Theta_grad= (  R .*(X*Theta'-Y)  )' * X + lambda .* Theta;

X_grad=     (  R .*(X*Theta'-Y)  ) * Theta + lambda .*X;

% h = X * Theta';
% diff = (h - Y);
% X_grad = (diff .* R) * Theta ;
% Theta_grad = (diff .* R)' * X ;

grad = [X_grad(:); Theta_grad(:)];

end
