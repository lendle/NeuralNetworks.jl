module SGD
import ..NeuralNetworks: AbstractSGD, update!

export AbstractSGD, SimpleSGD, AdaDelta, AdaGrad, AveragedSGD, update!

#should implement a update! method:
#update!(obj::AbstractSGD, weights::Vector{Float64}, gr::Vector{Float64})

#This function returns the vector of weights at which to evaluate the gradient
#It should be overridden by SGD implementations that need the gradient evaluated at
#values other than those stored in the Learner (AveragedSGD)
which_weights(obj::AbstractSGD, weights) = weights

type SimpleSGD{T} <: AbstractSGD{T}
    alpha1::T
    alpha2::T
    t::Int
    function SimpleSGD(alpha1::T, alpha2::T)
        alpha1 <= 0.0 && error("alpha1 should be positive")
        alpha2 < 0.0 && error("alpha2 should be non-negative")
        new(alpha1, alpha2, 0)
    end
end

function update!{T}(obj::SimpleSGD{T}, weights::Vector{T}, gr::Vector{T})
    obj.t += 1
    stepsize = - obj.alpha1 / (one(T) + obj.alpha1 * obj.alpha2 * obj.t)
    @simd for i in 1:length(gr) @inbounds weights[i] = gr[i] + stepsize end
    weights
end


type AdaDelta{T} <: AbstractSGD{T}
    rho::T
    eps::T
    sqgr::Vector{T}
    squp::Vector{T}
    up::Vector{T}
    initialized::Bool
    function AdaDelta(rho::T, eps::T)
        (rho <= 0.0 || eps <= 0.0) && error("rho and epsilon should be positive")
        obj = new()
        obj.rho = rho
        obj.eps = eps
        obj.initialized = false
        obj
    end
end

Base.show(io::IO, obj::AdaDelta) = print(io, "AdaDelta(ρ=$(obj.rho), ε=$(obj.eps))")

function init!(obj::AdaDelta, weights)
    obj.initialized && error("already initialized")
    obj.sqgr = zeros(weights)
    obj.squp = zeros(weights)
    obj.initialized = true
end

function update!{T}(obj::AdaDelta{T}, weights::Vector{T}, gr::Vector{T})
    obj.initialized || init!(obj, weights)
    @simd for i in 1:length(weights)
        gri = gr[i]
        @inbounds obj.sqgr[i] = obj.rho * obj.sqgr[i] + (one(T) - obj.rho) * gri * gri #line 4
        @inbounds upi = - sqrt(obj.squp[i] + obj.eps) / sqrt(obj.sqgr[i] + obj.eps) .* gri #line 5
        @inbounds obj.squp[i] = obj.rho * obj.squp[i] + (1.0 - obj.rho) * upi * upi #line 6
        @inbounds weights[i] =  weights[i] + upi #line 7
    end
    weights
end

type AdaGrad{T} <: AbstractSGD{T}
    eta::T
    sqgr::Vector{T}
    initialized::Bool
    function AdaGrad(eta::T)
        eta > 0.0 || error("eta should be positive")
        obj = new()
        obj.eta = eta
        obj.initialized = false
        obj
    end
end


Base.show(io::IO, obj::AdaGrad) = print(io, "AdaGrad(η=$(obj.eta))")

function init!{T}(obj::AdaGrad{T}, weights::Vector{T})
    obj.initialized && error("already initialized")
    obj.sqgr = fill!(similar(weights), convert(T, 1.0e-8))
    obj.initialized = true
end

function update!{T}(obj::AdaGrad, weights::Vector{T}, gr::Vector{T})
    obj.initialized || init!(obj, weights)
    @simd for i in 1:length(obj.sqgr)
        @inbounds gri = gr[i]
        @inbounds obj.sqgr[i] += gri * gri
        @inbounds weights[i] = obj.eta / sqrt(obj.sqgr[i]) * gr[i]
    end
    weights
end


## The averaged SGD needs the estimated gradient at different weights than those that
## you would use for prediction. In order to implement that, the train function
## could be modified to call which_weights on the AbstractSGD object, then
## compute the gradient at those weights. It's kind of weird... I'll leave it out for
## now.

# type AveragedSGD <: AbstractSGD
#     alpha1::Float64
#     alpha2::Float64
#     unaveraged_weights::Vector
#     t0::Int
#     t::Int
#     initialized::Bool
#     function AveragedSGD(alpha1::Float64, alpha2::Float64, t0::Int)
#         alpha1 <= 0.0 && error("alpha1 should be positive")
#         alpha2 < 0.0 && error("alpha2 should be non-negative")
#         obj = new()
#         obj.alpha1 = alpha1
#         obj.alpha2 = alpha2
#         obj.t0 = t0
#         obj.t = 0
#         obj.initialized = false
#         obj
#     end
# end

# Base.show(io::IO, obj::AveragedSGD) = print(io, "AveragedSGD(α1=$(obj.alpha1), α2=$(obj.alpha2), t0=$(obj.t0))")

# function init!(obj::AveragedSGD, weights)
#     obj.initialized && error("already initialized")
#     obj.unaveraged_weights = zeros(weights)
#     obj.initialized = true
# end

# which_weights(obj::AveragedSGD, weights) = obj.initialized? obj.unaveraged_weights: weights

# function update!(obj::AveragedSGD, weights::Vector{Float64}, gr::Vector{Float64})
#     obj.initialized || init!(obj, weights)
#     obj.t += 1
#     stepsize = - obj.alpha1 * (1.0 + obj.alpha1 * obj.alpha2 * obj.t)^-0.75
#     mu = 1.0 / max(1.0, obj.t - obj.t0)
#     asgdloop(obj.unaveraged_weights, stepsize, mu, gr, weights)
#     weights
# end
# function asgdloop(unaveraged_weights, stepsize, mu, gr, weights)
#     @simd for i in 1:length(weights)
#         @inbounds unaveraged_weights[i] = unaveraged_weights[i] + stepsize * gr[i]
#         @inbounds weights[i] = weights[i] + mu * (unaveraged_weights[i] .- weights[i])
#     end
# end

end

