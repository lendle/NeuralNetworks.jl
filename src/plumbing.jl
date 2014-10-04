# plumbing.jl implements some internal methods, iterator protocol for NNets, and
# show methods, etc.


## Show methods ##

Base.show{T <: InputLayer}(io::IO, layer::T) = print(io, "$T($(layer.nnodes))")
Base.show{T <: Layer}(io::IO, layer::T) = print(io, "$T($(layer.nnodes), $(layer.activation))")

function Base.show(io::IO, net::NNet)
    print(io, "NNet(")
    for layer in net
        show(io, layer)
        if !isoutput(layer)
            println(io, " →")
        end
    end
    print(io, ")")
end

## Layer utilities and connections

@doc """true if first layer in a network""" ->
isinput(::InputLayer) = true
isinput(::Layer) = false

@doc """true if last layer in a network""" ->
isoutput(::OutputLayer) = true
isoutput(::Layer) = false

@doc """true if the Layer/Connection is connected to other Parts""" ->
isconnected(layer::InputLayer) = isdefined(layer, :output)
isconnected(layer::OutputLayer) = isdefined(layer, :input)
function isconnected(obj::Layer)
    inconnected = isdefined(obj, :input)
    outconnected = isdefined(obj, :output)
    (inconnected $ outconnected) && error("Only one of input or output connected")
    inconnected && outconnected
end

@doc """connects the input and output Layers""" ->
function connect!(input::Layer, output::Layer)
    input.output = output
    output.input = input
end

@doc """returns the Part connected to the input of obj""" ->
prevlayer(layer::Layer) = layer.input

@doc """returns the Part connected to the output of obj""" ->
nextlayer(layer::Layer) = layer.output

### Iterators

Base.start(net::NNet) = (false, net.input)
function Base.next(::NNet, state)
    layer = state[2]
    if isa(layer, OutputLayer)
        return (layer, (true,))
    else
        return (layer, (false, nextlayer(layer)))
    end
end
Base.done(::NNet, state) = state[1]

type RevNetIter net::NNet end

#this docstring is causing issues
#@doc """iterate over the layers in an NNet from output to input""" ->
Base.reverse(net::NNet) = RevNetIter(net)
Base.start(iter::RevNetIter) = (false, iter.net.output)
function Base.next(::RevNetIter, state)
    layer = state[2]
    if isa(layer, InputLayer)
        return (layer, (true,))
    else
        return (layer, (false, prevlayer(layer)))
    end
end
Base.done(::RevNetIter, state) = state[1]

@doc """iterator of pairs of Layers in an NNet""" ->
layerpairs(net) = partition(net, 2, 1)

## Setters and getters for weights and gradient

@doc """Sets net's weights to Θ""" ->
setweights!{T}(net::NNet{T}, Θ::Vector{T}) = is(net.Θ, Θ) || copy!(net.Θ, Θ)

@doc """Sets Θ to net's weights""" ->
getweights!{T}(Θ::Vector{T}, net::NNet{T}) = is(net.Θ, Θ) || copy!(Θ, net.Θ)
@doc """Returns a copy of net's weights""" ->
getweights(net::NNet) = copy(net.Θ)

@doc """Sets Δ to net's gradient""" ->
getgradient!{T}(Δ::Vector{T}, net::NNet{T}) = is(net.Δ, Δ) || copy!(Δ, net.Δ)
@doc """Returns a copy of net's gradient""" ->
getgradient(net::NNet) = copy(net.Θ)

## Initialize weights

initializeweights!(layer::InputLayer) = layer

@doc """initialize the weights of a connection based on its ActivationFun""" ->
function initializeweights!{T}(layer::Layer{T})
    fanout, fanin = size(layer.Θ)
    @inbounds for i in 1:fanout layer.Θ₀[i] = init(layer.activation, fanin, fanout) end
    @inbounds for i in 1:length(layer.Θ)
        layer.Θ[i] = oftype(T, init(layer.activation, fanin, fanout))
    end
    layer
end


@doc """Reinitializes the weights for the whole network. If not already allocated,
storage for the weights and gradient are allocated, and views to them are constructed for each layer.
""" ->
function initializeweights!{T}(net::NNet{T})
    if !net.initialized
        num_params = 0
        for layer in drop(net, 1)
            num_params += (prevlayer(layer).nnodes + 1) * layer.nnodes
        end
        net.Θ = Array(T, num_params)
        net.Δ = Array(T, num_params)

        used = 0
        for layer in drop(net, 1)
            #contiguous_view is does not check bounds
            used + (prevlayer(layer).nnodes + 1) * layer.nnodes <= length(net.Θ) || error()
            layer.Θ₀ = contiguous_view(net.Θ, used, (layer.nnodes, 1))
            layer.Δ₀ = contiguous_view(net.Δ, used, (layer.nnodes, 1))
            used += layer.nnodes
            layer.Θ = contiguous_view(net.Θ, used, (layer.nnodes, prevlayer(layer).nnodes))
            layer.Δ = contiguous_view(net.Δ, used, (layer.nnodes, prevlayer(layer).nnodes))
            used += layer.nnodes * prevlayer(layer).nnodes
        end
        net.num_params = num_params
        net.initialized = true
    end

    for layer in net initializeweights!(layer) end
    net
end

## allocation of storage space

@doc """allocates storage space for n samples in a layer""" ->
function allocate{T}(layer::Layer{T}, n)
    layer.n == n && return layer
    layer.a = Array(T, layer.nnodes, n)
    if !isinput(layer) layer.δ = Array(T, layer.nnodes, n) end
    layer.n = n
    layer
end

@doc """allocates storage space for n samples. If storage has already been allocated
and it is the correct size, no additional allocations are performed.""" ->
function allocate(net::NNet, n)
    net.n == n && return net
    #don't need to allocate 'a' in the first layer, will use the data
    for layer in drop(net,1) allocate(layer, n) end
    net.n = n
    net
end

## apply activation fun ##

activate!(layer::InputLayer) = layer

@doc """applies the activation function to a Layer""" ->
function activate!(layer::Layer)
    activate!(layer.activation, layer.a)
end

function activate!{T}(activation::ActivationFun, a::DenseMatrix{T})
    @simd for i in 1:length(a)
        @inbounds a[i] = value(activation, a[i])
    end
end


## some tight inner loop functions

function calcderiv{T}(activation::ActivationFun, δ::DenseMatrix{T}, a::DenseMatrix{T})
    @simd for i in 1:length(δ)
        @inbounds δ[i] =  δ[i]* deriv(activation, a[i])
    end
end


function sumcost(activation::ActivationFun, a::DenseMatrix, y::DenseMatrix)
    c = 0.0
    @simd for i in 1:length(y)
        @inbounds c += cost(activation, a[i], y[i])
    end
    c
end
