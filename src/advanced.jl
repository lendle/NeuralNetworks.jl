# @doc """
# The methods exported by the Advaned module allow more manipulation of an NNet
# object and may mutate it in irreversible ways.
# The behavior of some methods may depend on the state of the NNet object. In particular
# the `backward` method calculates the gradient on the current mini-batch, but the results
# are meaningless if `forward` was not previously called on the same mini-batch.

# Methods with `_at` in the name compute some quantity at a particular value of the weights.
# They set the weight vector to whatever you want, so if you want to return the net to it's
# original state, you should save the previous weights with
# `getweights` and then reset them with `setweights`.
# """ ->
module Advanced
import ..NeuralNetworks: getweights, getweights!, getgradient!, setweights!,
    forward, backward, cost_at, gradient_at, gradient_at!, gradient, gradient!

export getweights, getweights!, getgradient!, setweights!,
    forward, backward, cost_at, gradient_at, gradient_at!, gradient, gradient!
end
