type Monitor
    train_costs::Vector{Float64}
    valid_costs::Vector{Float64}
    times::Vector{Float64}
    epochs::Int
    starttime::Float64

    maxtime::Float64
    maxepochs::Int

    patience::Int
    bestvalid_cost::Float64
    bestepoch::Int
    bestΘ::Vector
    function Monitor(;maxtime=0, maxepochs=0, patience=10)
        m = new()
        m.train_costs = Float64[]
        m.valid_costs = Float64[]
        m.times = Float64[]
        m.epochs = 0

        m.maxtime = maxtime
        m.maxepochs = maxepochs
        m.patience = patience

        m.bestvalid_cost = Inf
        m.bestepoch = 0
        m
    end
end

function start!(m::Monitor)
    m.starttime = time()
    m
end

function done(m::Monitor)
    m.epochs == 0 && return false
    m.maxtime > 0.0 && m.maxtime < m.times[end] && return true
    m.maxepochs > 0 && m.maxepochs <= m.epochs && return true
    m.bestepoch + m.patience < m.epochs && return true
    false
end

function update!(m::Monitor, train_cost, valid_cost, Θ)
    push!(m.times, time()-m.starttime)
    push!(m.train_costs, train_cost)
    push!(m.valid_costs, valid_cost)

    newbest = minimum(m.valid_costs)

    m.epochs += 1

    if newbest < m.bestvalid_cost
        m.bestvalid_cost = newbest
        m.bestepoch = m.epochs
        if isdefined(m, :bestΘ)
            copy!(m.bestΘ, Θ)
        else
            m.bestΘ = copy(Θ)
        end
    end
    println("Epoch: $(m.epochs), Total time: $(round(m.times[end], 2))s Train cost:$(round(train_cost,4)), Val cost:$(round(valid_cost,4))")
    m
end

type MiniBatchs
    xviews::Vector{DenseMatrix}
    yviews::Vector{DenseMatrix}
    nminibatches::Int
    function MiniBatchs(xviews, yviews)
        nminibatches = length(xviews)
        nminibatches == length(yviews) || error()
        mbs = new()
        mbs.xviews = xviews
        mbs.yviews = yviews
        mbs.nminibatches = nminibatches
        mbs
    end
end

function MiniBatchs(x::DenseMatrix, y::DenseMatrix, minibatchsize, warn=true)
    n = size(x,2)
    n == size(y, 2) || error("sizes of x and y do not match")
    rem(n, minibatchsize) == 0 && warn || error("Number of samples is not divisible buy minibatchsize, some samples will not be used")
    nminibatches = div(n, minibatchsize)
    xviews = DenseMatrix[]
    yviews = DenseMatrix[]
    for mb in 1:nminibatches
        start = (mb - 1) * minibatchsize + 1
        stop = mb * minibatchsize
        push!(xviews, view(x, :, start:stop))
        push!(yviews, view(y, :, start:stop))
    end
    MiniBatchs(xviews, yviews)
end

Base.start(mbs::MiniBatchs)  = 1
Base.next(mbs::MiniBatchs, state) = ((mbs.xviews[state], mbs.yviews[state]), state+1)
Base.done(mbs::MiniBatchs, state) = state > mbs.nminibatches

function train{T}(net::NNet{T}, x::DenseMatrix{T}, y::DenseMatrix{T}, x_val::DenseMatrix{T}, y_val::DenseMatrix{T}, minibatchsize::Int, monitor::Monitor, opt::AbstractSGD)
    mbs = MiniBatchs(x, y, minibatchsize)

    start!(monitor)

    valnet = deepcopy(net)
    valnet.Θ = net.Θ

    while !done(monitor)
        train_cost = 0.0
        for (x, y) in mbs
            train_cost += cost(net, x, y)
            backward(net, y)
            update!(opt, net.Θ, net.Δ)
        end
        train_cost /= mbs.nminibatches
        update!(monitor, train_cost, cost(valnet, x_val, y_val, false), net.Θ)
    end
    copy!(net.Θ, monitor.bestΘ)
    (net, monitor)
end
