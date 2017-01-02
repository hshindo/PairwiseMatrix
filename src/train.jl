using ProgressMeter

function train()
    train_x, train_y, test_x, test_y = setup_data()
    h5file = joinpath(dirname(@__FILE__), ".data/wordembeds_nyt100.h5")
    wordembeds = Lookup(h5read(h5file, "v"))
    model = setup_model(wordembeds)

    opt = SGD(0.00001)
    for epoch = 1:10
        println("epoch: $(epoch)")
        # opt.rate = 0.0075 / epoch
        prog = Progress(5000)
        loss = 0.0
        for i in randperm(5000)
            x, y = train_x[i], train_y[i]
            z = model(x)
            l = crossentropy(y, z)
            loss += mean(l.data)
            vars = gradient!(l)
            foreach(v -> opt(v.data,v.grad), vars)
            Merlin.update!(wordembeds, opt)

            next!(prog)
        end
        #loss = fit(train_x, train_y, model, crossentropy, opt)
        println("loss: $(loss)")
    end
end
