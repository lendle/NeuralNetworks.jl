module NeuralNetworks
using Docile
@docstrings
using Iterators, ArrayViews, NumericExtensions, NumericFuns

export nnet, cost, predict, predict!
export Sigmoid, SoftPlus, Linear, ReLU, Tanh, SoftSign
export Advanced

import Base.gradient



include("types.jl")
include("activations.jl") #defines acivation functions
include("plumbing.jl") #implements some internals

function nnet(layersizes::Vector{Int}, activations=[Sigmoid()]::Vector, λ=0.0, T=Float64)
    #check inputs
    length(layersizes) > 1 || error("layersizes should have length at least 2")
    if length(activations) == 1
        activations = fill(activations[1], length(layersizes) - 1)
    end
    length(layersizes) - length(activations) == 1 ||
        error("Should specify one activation, or one per layer excluding the input layer")

    lastactivation = activations[end]
    method_exists(cost, (typeof(lastactivation), T,T)) ||
        error("$(lastactivation) cannot be used as the activation function for the output layer")
    #done checking inputs

    λ = convert(T, λ)

    #build and connect layers
    input = InputLayer{T}(layersizes[1])
    output = input
    for (nnodes, act) in zip(layersizes[2:end-1], activations[1:end-1])
        temp = output
        output = HiddenLayer{T}(nnodes, act, λ)
        connect!(temp, output)
    end
    temp = output
    output = OutputLayer{T}(layersizes[end], lastactivation, λ)
    connect!(temp, output)

    net = NNet{T}(input, output, λ)
    initializeweights!(net)
    net
end

@doc """performs forward pass on input x""" ->
function forward(net::NNet, x::Matrix)
    p, n = size(x)
    net.input.nnodes == p || error("x and nnodes in the input layer do not match")
    net.input.a=x
    allocate(net, n)
    for (in, output) in layerpairs(net)
        forward(in, output)
    end
end

@doc """performs a step in the forward pass between the given input and output layers""" ->
function forward(input::Layer, output::Layer)
    A_mul_B!(output.a, output.Θ, input.a)
    vbroadcast1!(Add(), output.a, output.Θ₀, 1)
    #@error() #need to broadcast here add1!(output.a, con.Θ₀)
    activate!(output)
end

@doc """performs the backwards pass given outcomes y""" ->
function backward{T}(net::NNet{T}, y::DenseMatrix{T})
    p, n = size(y)
    net.output.nnodes == p || error("y and nnodes in the output layer do not match")
    net.n == n || error("incorrect number of samples in y")

    map!(Subtract(), net.output.δ, net.output.a, y)
    for (output, input) in layerpairs(reverse(net))
        # backward(input, output)
        ## for some reason if the contents of this loop are put into their own function
        ## julia segfaults on the copy!(output.Δ, output.Θ) line
        ## This seems to have to do with the views of Δ and Θ
        ## Specifically, when passed to the loop body function, output.Δ seems to lose
        ## the pointer of it's parent array. pointer(output.Δ) works... well it returns
        ## a number at least, but pointer(output.Δ.arr) causes a segfault.
        n = size(output.δ, 2)
        mean!(output.Δ₀, output.δ, 2)
        copy!(output.Δ, output.Θ)
        BLAS.gemm!('N', 'T', one(T)/n, output.δ, input.a, output.λ, output.Δ)
        if !isinput(input)
            At_mul_B!(input.δ, output.Θ, output.δ)
            calcderiv(input.activation, input.δ, input.a)
        end
    end
end

function cost_at{T}(net::NNet{T}, Θ::Vector{T}, x::DenseMatrix{T}, y::DenseMatrix{T}, penalty=true)
    setweights!(net, Θ)
    forward(net, x)
    lastactivation = net.output.activation
    c = sumcost(lastactivation, net.output.a, y)/size(y,2)

    if penalty
        pen = 0.0
        for layer in drop(net, 1)
            pen += layer.λ * sumsq(layer.Θ) / 2
        end
        c += pen
    end
    isnan(c)? inf(T) : c
end

cost(net::NNet, x, y, penalty=true) = cost_at(net, net.Θ, x, y, penalty)

function gradient_at!{T}(net::NNet{T}, Δ::Vector{T}, Θ::Vector{T}, x::DenseMatrix{T}, y::DenseMatrix{T})
    setweights!(net, Θ)
    forward(net, x)
    backward(net, y)
    getgradient!(Δ, net)
    Δ
end
gradient!(net::NNet, Δ::Vector, x::DenseMatrix, y::DenseMatrix) = gradient_at!(net, Δ, net.Θ, x, y)
gradient!(net::NNet, x::DenseMatrix, y::DenseMatrix) = gradient!(net, net.Δ, x, y)
gradient_at(net::NNet, Θ::Vector, x::DenseMatrix, y::DenseMatrix) = gradient!(net, similar(Θ), x, y)
gradient(net::NNet, x::DenseMatrix, y::DenseMatrix) = gradient_at(net, net.Θ, x, y)

function predict!{T}(net::NNet{T}, p::DenseMatrix{T}, x::DenseMatrix{T})
    forward(net, x)
    copy!(p, net.output.a)
end

function predict{T}(net::NNet{T}, x::DenseMatrix{T})
    forward(net, x)
    copy(net.output.a)
end

include("advanced.jl")

end # module
