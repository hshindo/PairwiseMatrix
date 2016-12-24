type Token
    word::Int
    chars::Vector{Int}
end

function setup_data()
    h5file = joinpath(dirname(@__FILE__), ".data/wordembeds_nyt100.h5")
    words = h5read(h5file, "s")
    wordvecs = h5read(h5file, "v")
    worddict = Dict(words[i] => i for i=1:length(words))
    chardict = Dict(' ' => 1)

    #traindata = UD_English.traindata()
    #testdata = UD_English.testdata()
    traindata = CoNLL.read(joinpath(dirname(@__FILE__), ".data/wsj_00-18.conll"))
    testdata = CoNLL.read(joinpath(dirname(@__FILE__), ".data/wsj_22-24.conll"))
    info("# sentences of train data: $(length(traindata))")
    info("# sentences of test data: $(length(testdata))")

    train_x, train_y = setup_xy(traindata, worddict, chardict, true)
    test_x, test_y = setup_xy(testdata, worddict, chardict, false)
    info("# words: $(length(worddict))")
    info("# chars: $(length(chardict))")
    train_x, train_y, test_x, test_y
end

function setup_xy(doc, worddict, chardict, append)
    data_x, data_y = Vector{Token}[], Vector{Int}[]
    unkword = worddict["UNKNOWN"]
    for sent in doc
        push!(data_x, Token[])
        push!(data_y, Int[])
        for items in sent
            word = items[2]
            word0 = replace(word, r"[0-9]", '0')
            wordid = get(worddict, lowercase(word0), unkword)
            chars = Vector{Char}(word0)
            if append
                charids = map(c -> get!(chardict,c,length(chardict)+1), chars)
            else
                charids = map(c -> get(chardict,c,0), chars)
            end
            head = parse(Int, items[7])
            token = Token(wordid, charids)
            push!(data_x[end], token)
            push!(data_y[end], head)
        end
    end
    data_x, data_y
end
