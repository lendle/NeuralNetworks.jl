# plumbing.jl implements some internal methods, iterator protocol for NNets, and
# show methods, etc.


Base.show{T <: InputLayer}(io::IO, layer::T) = print(io, "$T($(layer.nnodes))")
Base.show{T <: Layer}(io::IO, layer::T) = print(io, "$T($(layer.nnodes), $(layer.activation))")
#Base.show(io::IO, con::Connection) = show(io, typeof(con.activation))

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
prev(obj::Layer) = obj.input

#@doc """returns the Part connected to the output of obj""" ->
next(obj::Layer) = obj.output


@doc """initialize the weights of a connection based on its ActivationFun""" ->
function initializeweights!{T}(layer::Layer{T})
    fanout, fanin = size(layer.Θ)
    @inbounds for i in 1:fanout layer.Θ₀[i] = init(layer.activation, fanin, fanout) end
    @inbounds for i in 1:length(layer.Θ)
        layer.Θ[i] = oftype(T, init(layer.activation, fanin, fanout))#2 * c * rand(T) - c
    end
end

@doc """allocates storage space for n samples""" ->
function allocate!{T}(layer::Layer{T}, n)
    layer.a = Array(T, layer.nnodes, n)
    if !isinput(layer) layer.δ = Array(T, layer.nnodes, n) end
    layer
end

### Iterators

Base.start(net::NNet) = (false, net.input)
function Base.next(::NNet, state)
    layer = state[2]
    if isa(layer, OutputLayer)
        return (layer, (true,))
    else
        return (layer, (false, next(layer)))
    end
end
Base.done(::NNet, state) = state[1]

type RevNetIter net::NNet end

#@doc "Iterate over the Parts of an NNet from output to input" ->
Base.reverse(net::NNet) = RevNetIter(net)
Base.start(iter::RevNetIter) = (false, iter.net.output)
function Base.next(::RevNetIter, state)
    layer = state[2]
    if isa(layer, InputLayer)
        return (layer, (true,))
    else
        return (layer, (false, prev(layer)))
    end
end
Base.done(::RevNetIter, state) = state[1]

@doc """iterator of pairs of Layers""" ->
layerpairs(net) = partition(net, 2, 1)


@doc """applies the activation function to a Layer""" ->
function activate!(layer::Layer)
    isinput(layer) && error("Can't activate the input layer")
    activate!(layer.activation, layer.a)
end

function activate!{T}(activation::ActivationFun, a::DenseMatrix{T})
    @simd for i in 1:length(a)
        @inbounds a[i] = value(activation, a[i])
    end
end

@doc """Reinitializes the weights for the whole network. If not already allocated,
storage for the weights and gradient are allocated, and views to them are constructed
in Connections.
""" ->
function initializeweights!{T}(net::NNet{T})
    if !net.initialized
        num_params = 0
        for layer in drop(net, 1)
            num_params += (layer.input.nnodes + 1) * layer.nnodes
        end
        net.Θ = Array(T, num_params)
        net.Δ = Array(T, num_params)

        used = 0
        for layer in drop(net, 1)
            #contiguous_view is does not check bounds
            used + (layer.input.nnodes + 1) * layer.nnodes <= length(net.Θ) || error()
            layer.Θ₀ = contiguous_view(net.Θ, used, (layer.nnodes, 1))
            layer.Δ₀ = contiguous_view(net.Δ, used, (layer.nnodes, 1))
            used += layer.nnodes
            layer.Θ = contiguous_view(net.Θ, used, (layer.nnodes, layer.input.nnodes))
            layer.Δ = contiguous_view(net.Δ, used, (layer.nnodes, layer.input.nnodes))
            used += layer.nnodes * layer.input.nnodes
        end
        net.num_params = num_params
        net.initialized = true
    end

    for layer in drop(net, 1) initializeweights!(layer) end
end

@doc """allocates storage space for n samples. If storage has already been allocated
and it is the correct size, no additional allocations are performed.""" ->
function allocate!(net::NNet, n)
    n == net.n && return net
    #don't need to allocate 'a' in the first layer, will use the data
    for layer in drop(net,1) allocate!(layer, n) end
    net.n = n
end

@doc """Sets net's weights to Θ""" ->
setΘ!{T}(net::NNet{T}, Θ::Vector{T}) = is(net.Θ, Θ) || copy!(net.Θ, Θ)
@doc """Sets Θ to net's weights""" ->
getΘ!{T}(Θ::Vector{T}, net::NNet{T}) = is(net.Θ, Θ) || copy!(Θ, net.Θ)
@doc """Sets Δ to net's weights""" ->
getΔ!{T}(Δ::Vector{T}, net::NNet{T}) = is(net.Δ, Δ) || copy!(Δ, net.Δ)

getΘ(net::NNet) = copy(net.Θ)

function sumcost(activation::ActivationFun, a::DenseMatrix, y::DenseMatrix)
    c = 0.0
    @simd for i in 1:length(y)
        @inbounds c += cost(activation, a[i], y[i])
    end
    c
end
