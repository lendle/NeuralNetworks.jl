using NeuralNetworks
using Base.Test

include("activations.jl")

#this is kind of a stupid test, but if it passes, then all the major functionality
#is either working, or completely broken
using NeuralNetworks, NeuralNetworks.Advanced, Calculus
x = rand(10, 100)
yy = iceil(rand(100)* 3)
y = zeros(3, 100)
for (j, i) in enumerate(yy)
    y[i,j] = 1
end
y
net = nnet([10,6, 7, 10,  4, 5, 3], [Linear(), SoftPlus(), Tanh(), ReLU(), Sigmoid(), SoftMax()], 0.0)
#net = NNet(rand(10, 100), round(rand(2, 100)), [10, 4, 10, 7, 9, 2], 0.0);
cc(Θ) = cost_at(net, Θ, x, y)
Θ = getweights(net)

@test maxabs(extrema(NeuralNetworks.gradient_at(net, Θ, x, y) .- derivative(cc, Θ))) < 1e-8
