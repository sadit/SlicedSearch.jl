using SimilaritySearch, HDF5, JLD2, LinearAlgebra, SlicedSearch, DataFrames, CSV


function laion_benchmark()
    @time X = load("data/laion2B-en-clip768v2-n=300K.h5", "emb")
    @time Q = load("data/public-queries-10k-clip768v2.h5", "emb")
    # Q = Q[:, 1:128]
    for c in eachcol(X)
        normalize!(c)
    end
    for c in eachcol(Q)
        normalize!(c)
    end
    @show size(X), size(Q)
    StrideMatrixDatabase(X), StrideMatrixDatabase(Q), NormalizedCosineDistance()
end

function main_split(dist, db, queries, k, slice, D, Igold,
        npartslist=(1.2, 1.5, 2f0, 3f0, 4f0),
        maxsplitlist=[4, 8, 12]
    )
    @info "evaluating $slice with FixedSplit"
    for nparts in npartslist
        idx = SlicedEvaluation(dist, db, slice, FixedSplit(nparts))
        t = @elapsed Is, Ds = searchbatch(idx, queries, k)
        r = macrorecall(Igold, Is)
        if slice isa FixedSlice
            push!(D, ("SlicedEvaluation", "F", slice.Δ, "F", nparts, r, t))
        else
            push!(D, ("SlicedEvaluation", "D", slice.maxΔ, "F", nparts, r, t))
        end
    end

    @info "evaluating $slice with DecreasingSplit"
    for maxsplit in maxsplitlist
        minsplit = 2
        idx = SlicedEvaluation(dist, db, slice, DecreasingSplit(minsplit, maxsplit))
        t = @elapsed Is, Ds = searchbatch(idx, queries, k)
        r = macrorecall(Igold, Is)
        if slice isa FixedSlice
            push!(D, ("SlicedEvaluation", "F", slice.Δ, "D", maxsplit, r, t))
        else
            push!(D, ("SlicedEvaluation", "D", slice.maxΔ, "D", maxsplit, r, t))
        end
    end
end

function main(db, queries, dist)
    k = 16
    output = "laion-300k-queries-10k-k=16.csv"
    E = ExhaustiveSearch(; db, dist)
    t = @elapsed Igold, Dgold = searchbatch(E, queries, k)
    @show size(Igold)

    @info "now running the algorithm"
    D = DataFrame(name=String[], slicetype=String[], slicearg=Float32[], splittype=String[], splitarg=Float32[], recall=Float32[], searchtime=Float32[])
    push!(D, ("exhaustive", "", 0, "", 0, 1, t))

    for Δ in (4, 6, 8, 10, 12)
        slice = FixedSlice(Δ)
        main_split(dist, db, queries, k, slice, D, Igold)
    end

    for maxΔ in (10, 12, 14, 16, 20)
        slice = DecreasingSlice(; maxΔ)
        main_split(dist, db, queries, k, slice, D, Igold)
    end

    CSV.write(output, D)
    D
end

function main_aknn(db, dist)
    k = 16
    output = "laion-300k-allknn-k=16.csv"
    E = ExhaustiveSearch(; db, dist)
    t = @elapsed Igold, Dgold = searchbatch(E, db, k)
    @show size(Igold)

    @info "now running the algorithm"
    D = DataFrame(name=String[], slicetype=String[], slicearg=Float32[], splittype=String[], splitarg=Float32[], recall=Float32[], searchtime=Float32[])
    push!(D, ("exhaustive", "", 0, "", 0, 1, t))

    for Δ in (8,)
        slice = FixedSlice(Δ)
        main_split(dist, db, db, k, slice, D, Igold, (3,), (12,))
    end

    for maxΔ in (16,)
        slice = DecreasingSlice(; maxΔ)
        main_split(dist, db, db, k, slice, D, Igold, (3,), (12,))
    end

    CSV.write(output, D)
    D
end


function create_searchgraph(indexfile, db, dist)
    @info "creating $indexfile"

    G
end

function main_searchgraph(db, queries, dist)
    k = 16
    output = "laion-300k-searchgraph-k=16-queries-10k.csv"
    outputallknn = "laion-300k-searchgraph-k=16-allknn.csv"
    E = ExhaustiveSearch(; db, dist)
    t = @elapsed Igold, Dgold = searchbatch(E, db, k)
    @show size(Igold)

    @info "now running the algorithm"
    D = DataFrame(name=String[], buildtime=Float32[], optimtime=Float32[], recall=Float32[], searchtime=Float32[])
    Da = DataFrame(name=String[], buildtime=Float32[], optimtime=Float32[], recall=Float32[], searchtime=Float32[])
    G = SearchGraph(; db, dist, verbose=false)
    callbacks = SearchGraphCallbacks(MinRecall(0.9), verbose=false)
    neighborhood = Neighborhood(logbase=1.5)
    buildtime = @elapsed index!(G; neighborhood, callbacks)

    for minrecall in (0.7, 0.8, 0.9, 0.95, 0.99)
        optimtime = @elapsed = optimize!(G, MinRecall(minrecall))
        I, _ = searchbatch(G, queries, k)
        recall = macrorecall(Igold, I)
        push!(D, ("ABS r=$minrecall", buildtime, optimtime, recall, searchtime))

        I, _ = searchbatch(G, database(G), k)
        recall = macrorecall(Igold, I)
        push!(Da, ("ABS r=$minrecall", buildtime, optimtime, recall, searchtime))
    end

    CSV.write(output, D)
    CSV.write(outputallknn, Da)
    D, Da
end
