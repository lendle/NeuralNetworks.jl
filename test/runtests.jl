using NeuralNetworks
using Base.Test

include("activations.jl")

#this is kind of a stupid test, but if it passes, then all the major functionality
#is either working, or completely broken
using NeuralNetworks, NeuralNetworks.Advanced, Calculus
x = rand(10, 100)
y = round(rand(3, 100))
net = nnet([10,6, 7, 10,  4, 3], [Linear(), SoftPlus(), Tanh(), ReLU(), Sigmoid()], 0.0)
#net = NNet(rand(10, 100), round(rand(2, 100)), [10, 4, 10, 7, 9, 2], 0.0);
cc(Θ) = cost_at(net, Θ, x, y)
Θ = getweights(net)

@test maxabs(extrema(NeuralNetworks.gradient_at(net, Θ, x, y) .- derivative(cc, Θ))) < 1e-8
