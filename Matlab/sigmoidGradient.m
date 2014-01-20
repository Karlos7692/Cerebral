
function g = sigmoidGradient(z)

g = zeros(size(z));
g = sigmoid(z) .* (ones(size(z)) - sigmoid(z));

end
