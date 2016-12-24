function setup_model()
    T = Float32
    x = Var()
    y = Lookup(x)
    y = pairwise(y)
    y = Conv(T, (3,3,200,128), padding=1)
    y = pooling(:max, (2,2), strides=2)
    compile(y, x)
end

function forward(m, data::Vector{Token})
    wordvec = map(t -> t.word, tokens)
    wordvec = reshape(wordvec, 1, length(wordvec))
    wordmat = m.wordfun(Var(wordvec))

    charvecs = map(tokens) do t
        #Var(zeros(Float32,50,1))
        charvec = reshape(t.chars, 1, length(t.chars))
        m.charfun(Var(charvec))
    end
    charmat = concat(2, charvecs)
    m.sentfun(wordmat, charmat)
end
