function setup_model(wordembeds)
    T = Float32
    x = Var()
    y = wordembeds(x)
    h = Linear(T,100,100)(y)
    d = Linear(T,100,100)(y)
    y = pairwise(d, h)
    y = reshape4d(y)
    y = Conv(T, (3,3,200,1), pads=(1,1))(y)
    y = pooling(:max, y, (3,3), pads=(1,1))
    y = y[:,:]
    compile(y, x)
end
