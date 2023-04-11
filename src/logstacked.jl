using SimilaritySearch, JLD2, LinearAlgebra
using SimilaritySearch.AdjacencyLists


const CacheP = [Vector{IdWeight}(undef, 32)]

function __init__()
    while length(CacheP) < Threads.nthreads()
        push!(CacheP, copy(CacheP[1]))
    end
end

function reuse_cache!(n::Int)
    P = CacheP[Threads.threadid()]
    P = resize!(P, n)
    @inboiunds for i in eachindex(P)
        P[i] = IdWeight(i, 0f0)
    end
    
    P    
end

function search_by_layers(dist, X, qvec, P, stop_at, Δ, nparts)
    dim = size(X, 1)
    b = ceil(Int, Δ*log(2, dim))
    sp = 1
    ep = min(dim, sp + b - 1)
    #s = 0
    @inbounds while length(P) > stop_at && sp <= dim
        # s += 1
        q = view(qvec, sp:ep)
        for (i, p) in enumerate(P)
            v = view(X, sp:ep, p.id)
            P[i] = IdWeight(p.id, p.weight + evaluate(dist, q, v))
        end

        n = ceil(Int, length(P) / nparts)
        partialsort!(P, 1:n, by=p->p.weight)
        resize!(P, n)
        sp = ep + 1
        ep = min(dim, sp + b - 1)
    end
    
    P
end

struct LogStackedSearch{DistType<:SemiMetric,DBType<:AbstractDatabase} <: AbstractSearchIndex
    dist::DistType
    db::DBType
    Δ::Int
    nparts::Float32
end

copy(idx::LogStackedSearch; dist=idx.dist, db=idx.db, Δ=idx.Δ, nparts=idx.nparts) = LogStackedSearch(dist, db, Δ, nparts)

function SimilaritySearch.searchbatch(idx::LogStackedSearch, queries::AbstractDatabase, k::Integer; pools=nothing)
    X = idx.db.matrix
    n = size(X, 2)
    I = Matrix{Int32}(undef, k, n)
    D = Matrix{Float32}(undef, k, n)
    GC.enable(false)
    
    Threads.@threads for i in eachindex(queries)
        P = reuse_cache!(n)
        P = search_by_layers(idx.dist, X, queries[i], P, idx.nparts * k, idx.Δ)
        
        @inbounds for j in 1:k
            I[j, i] = P[j].id
        end
        
        @inbounds for j in 1:k
            D[j, i] = P[j].weight
        end
    end
    GC.enable(true)
    
    I, D
end

function SimilaritySearch.search(idx::LogStackedSearch, q, res::KnnResult; pools=nothing)
    X = idx.db.matrix
    GC.enable(false)
    
    P = reuse_cache!(n)
    P = search_by_layers(idx.dist, X, queries[i], P, idx.nparts * k, idx.Δ)
    
    for i in 1:maxlength(resize)
        push!(res, P[i])
    end
    
    SearchResult(res, 0)
end
