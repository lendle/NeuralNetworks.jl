module NeuralNetworks
using Docile
@docstrings
using Iterators, ArrayViews, NumericExtensions, NumericFuns

export NNet, Layer, Connection

export Sigmoid, SoftPlus, Linear, ReLU, Tanh

export forward, backward

export cost, gradient, gradient!

export getΘ!, getΘ, getΔ!, setΘ!, initializeweights!

include("types.jl")
include("activations.jl") #defines acivation functions
include("plumbing.jl") #implements some internals

function NNet(layersizes::Vector{Int}, activations=[Sigmoid()]::Vector, λ=0.0, T=Float64)
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

    NNet{T}(input, output, λ)
end

@doc """performs forward pass on input x""" ->
function forward(net::NNet, x::Matrix)
    p, n = size(x)
    net.input.nnodes == p || error("x and nnodes in the input layer do not match")
    net.input.a=x
    allocate!(net, n)
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

function calcderiv{T}(activation::ActivationFun, δ::DenseMatrix{T}, a::DenseMatrix{T})
    @simd for i in 1:length(δ)
        @inbounds δ[i] =  δ[i]* deriv(activation, a[i])
    end
end


## this function causes a segfault, no idea why.
## the code above is identical, just not in it's own function
# function backward{T}(input::Layer{T}, output::Layer{T})
#     con = connection(output, input)
#     n = size(output.δ, 2)
#     mean!(con.Δ₀, output.δ, 2)
#     BLAS.gemm!('N', 'T', one(T)/n, output.δ, input.a, zero(T), con.Δ)
#     if !isinput(input)
#         At_mul_B!(input.δ, con.Θ, output.δ)
#         @inbounds for i in 1:length(input.δ)
#             input.δ[i] = input.δ[i] * deriv(con.activation, input.a[i])
#         end
#     end
# end

function cost{T}(net::NNet{T}, Θ::Vector{T}, x::DenseMatrix{T}, y::DenseMatrix{T}, penalty=true)
    setΘ!(net, Θ)
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

cost(net::NNet, x, y, penalty=true) = cost(net, net.Θ, x, y, penalty)

function gradient!{T}(net::NNet{T}, Δ::Vector{T}, Θ::Vector{T}, x::DenseMatrix{T}, y::DenseMatrix{T})
    setΘ!(net, Θ)
    forward(net, x)
    backward(net, y)
    getΔ!(Δ, net)
    Δ
end
gradient!(net::NNet, Δ::Vector, x::DenseMatrix, y::DenseMatrix) = gradient!(net, Δ, net.Θ, x, y)
gradient!(net::NNet, x::DenseMatrix, y::DenseMatrix) = gradient!(net, net.Δ, x, y)
gradient(net::NNet, Θ::Vector, x::DenseMatrix, y::DenseMatrix) = gradient!(net, similar(Θ), x, y)
gradient(net::NNet, x::DenseMatrix, y::DenseMatrix) = gradient(net, net.Θ, x, y)

function predict!{T}(net::NNet{T}, p::DenseMatrix{T}, x::DenseMatrix{T})
    forward(net, x)
    copy!(p, net.output.a)
end

function predict{T}(net::NNet{T}, x::DenseMatrix{T})
    forward(net, x)
    copy(net.output.a)
end

end # module
