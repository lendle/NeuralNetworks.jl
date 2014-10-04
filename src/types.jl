@doc """
an ActivationFun should implement a value method, and a deriv method.
Can also implement a const function, if the the activation is appropriate for the last layer.

Note that the deriv method should actually be the derivative of the activation
function w.r.t. it's input in terms of it's output. E.g.:
sigmoid(x) = 1/(1+exp(-x))
\frac{d}{dx} sigmoid(x) = 1/(1+exp(-x)) * (1-1/(1+exp(-x))) = sigmoid(x) .*  (1 - sigmoid(x))
so deriv(::Sigmoid, x) = x * (1-x)

I hope this is a reasonable thing to do...
""" ->
abstract ActivationFun

abstract Optimizer

@doc """A Layer represents the nodes in the network
and holds storage for the forward and backwards passes""" ->
abstract Layer{T <: FloatingPoint} #<: Part{T}

type InputLayer{T} <: Layer{T}
    nnodes::Int
    output::Layer{T}
    a::Matrix{T}
    function InputLayer(nnodes)
        layer=new()
        layer.nnodes=nnodes
        layer
    end
end

type OutputLayer{T} <: Layer{T}
    nnodes::Int
    input::Layer{T}
    a::Matrix{T}
    δ::Matrix{T}
    activation::ActivationFun
    Θ₀::DenseMatrix{T} #weights for bias term
    Θ::DenseMatrix{T} #weights
    Δ₀::DenseMatrix{T} #grad (after a backward pass) for bias term
    Δ::DenseMatrix{T} #grads
    λ::T
    n::Int
    function OutputLayer(nnodes, activation::ActivationFun, λ::T)
        layer=new()
        layer.nnodes = nnodes
        layer.activation = activation
        layer.λ = λ
        layer.n = 0
        layer
    end
end

type HiddenLayer{T} <: Layer{T}
    nnodes::Int
    input::Layer{T}
    output::Layer{T}
    a::Matrix{T}
    δ::Matrix{T}
    activation::ActivationFun
    Θ₀::DenseMatrix{T} #weights for bias term
    Θ::DenseMatrix{T} #weights
    Δ₀::DenseMatrix{T} #grad (after a backward pass) for bias term
    Δ::DenseMatrix{T} #grads
    λ::T
    n::Int
    function HiddenLayer(nnodes, activation::ActivationFun, λ::T)
        layer=new()
        layer.nnodes = nnodes
        layer.activation = activation
        layer.λ = λ
        layer.n = 0
        layer
    end
end


@doc """
Holds pointers to the input and output layers of the network,
the weight vector θ, and the gradient vector Δ.
Θ and Δ should never be reassigned because the Connections between layers hold
views into them.
""" ->
type NNet{T}
    input::Layer{T}
    output::Layer{T}
    Θ::Vector{T} #weights
    Δ::Vector{T} #gradient (after backwards pass)
    λ::T #l2 regularization
    n::Int #sample size of the currently allocated
    num_params::Int #total number of weights, should be == length(θ) == length(Δ)
    initialized::Bool
    function NNet(input, output, λ)
        net = new()
        net.input = input
        net.output = output
        net.λ = λ
        net.n = 0
        net.initialized = false
        net
    end
end
