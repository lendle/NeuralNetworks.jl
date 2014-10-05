function activate!{T}(activation::ActivationFun, a::DenseMatrix{T})
    @simd for i in 1:length(a)
        @inbounds a[i] = value(activation, a[i])
    end
end

#generic initializer because I don't know what I'm doing
init(::ActivationFun, fanin, fanout) = (c = sqrt(6/(fanin + fanout)); c*(2*rand()-1))

type Sigmoid <: ActivationFun end

value(::Sigmoid, x) = one(x)/(one(x)+exp(-x))
deriv(::Sigmoid, x) = x * (one(x)-x)
cost{T}(::Sigmoid, x::T, y::T) = y == one(T) ? -log(x) :
                                 y == zero(T) ? - log(one(T) - x) :
                                                - y * log(x) - (1-y) * log(one(T)-x)
#http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2010_GlorotB10.pdf (16)
init(::Sigmoid, fanin, fanout) = (c = 4*sqrt(6/(fanin + fanout)); c*(2*rand()-1))

type Tanh <: ActivationFun end
value(::Tanh, x) = tanh(x)
deriv(::Tanh, x) = one(x) - x*x
init(::Tanh, fanin, fanout) = (c = sqrt(6/(fanin + fanout)); c*(2*rand()-1))

type SoftPlus <: ActivationFun end
value(::SoftPlus, x) = log(1 .+ exp(x))
deriv(::SoftPlus, x) = one(x) - exp(-x)# 1 ./(1 .+ exp(-x))

type ReLU <: ActivationFun end
value(::ReLU, x) = max(zero(x), x)
deriv(::ReLU, x) = x == zero(x) ? zero(x) : one(x)
init(::ReLU, fanin, fanout) = (c = sqrt(6/(fanin + fanout)); c*(rand()))

type Linear <: ActivationFun end
value(::Linear, x) = x
deriv(::Linear, x) = one(x)
cost(::Linear, x, y) = (z = x-y; z*z/2)

type SoftMax <: ActivationFun end
deriv(::SoftMax, x) = x * (one(x)-x)
cost{T}(::SoftMax, x::T, y::T) = y == one(T) ? - log(x) : zero(T)

function activate!{T}(activation::SoftMax, a::DenseMatrix{T})
    for j in 1:size(a, 2)
        lsea = zero(T) #lsea for log(sum(exp(a)))
        for i in 1:size(a, 1)
            lsea += exp(a[i, j])
        end
        lsea = log(lsea)
        for i in 1:size(a, 1)
            a[i,j] = exp(a[i, j] - lsea)
        end
    end
    a
end
