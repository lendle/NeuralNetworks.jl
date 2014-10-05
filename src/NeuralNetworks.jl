module NeuralNetworks
using Docile
@docstrings
using Iterators, ArrayViews, NumericExtensions, NumericFuns, Reexport

export nnet, cost, predict, predict!, train
export Sigmoid, SoftPlus, Linear, ReLU, Tanh, SoftMax
export Advanced
import Base.gradient



include("types.jl")
include("activations.jl") #defines acivation functions
include("plumbing.jl") #implements some internals
include("nnet.jl") #implements main forward backward
include("train.jl") #training loop
include("sgd.jl") #implements a few mini-batch methods
@reexport using .SGD
include("advanced.jl")

end # module
