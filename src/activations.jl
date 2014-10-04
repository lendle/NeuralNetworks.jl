#generic initializer because I don't know what I'm doing
init(::ActivationFun, fanin, fanout) = randn()/100.0

type Sigmoid <: ActivationFun end

value(::Sigmoid, x) = one(x)/(one(x)+exp(-x))
deriv(::Sigmoid, x) = x * (one(x)-x)
cost(::Sigmoid, x, y) = - y * log(x) - (1-y) * log(1-x)
#http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2010_GlorotB10.pdf (16)
init(::Sigmoid, fanin, fanout) = (c = sqrt(6/(fanin + fanout)); c*(2*rand()-1))

type Tanh <: ActivationFun end
value(::Tanh, x) = tanh(x)
deriv(::Tanh, x) = one(x) - x*x
init(::Tanh, fanin, fanout) = (c = 4*sqrt(6/(fanin + fanout)); c*(2*rand()-1))

type SoftPlus <: ActivationFun end
value(::SoftPlus, x) = log(1 .+ exp(x))
deriv(::SoftPlus, x) = one(x) - exp(-x)# 1 ./(1 .+ exp(-x))

type ReLU <: ActivationFun end
value(::ReLU, x) = max(zero(x), x)
deriv(::ReLU, x) = x == zero(x) ? zero(x) : one(x)

type Linear <: ActivationFun end
value(::Linear, x) = x
deriv(::Linear, x) = one(x)
cost(::Linear, x, y) = (z = x-y; z*z/2)
