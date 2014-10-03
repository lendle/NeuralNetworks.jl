module TestActivations
import NeuralNetworks: value, deriv
using Base.Test, Calculus, NeuralNetworks

z = -1.0:0.1:1.0

for act in [Sigmoid(), ReLU(), SoftPlus(), Linear(), Tanh()]
    println("Testing $act")
    f(x) = value(act, x)

    for x in z
        if isa(act, ReLU) && x == 0.0
            @test deriv(act, value(act, x)) == 0.0
        else
            @test_approx_eq_eps(f'(x), deriv(act, value(act, x)), 1e-8)
        end
    end
end
end
