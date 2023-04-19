using SimilaritySearch, JLD2, LinearAlgebra
using SimilaritySearch.AdjacencyLists

export searchbatch, search, SlicedEvaluation, FixedSplit, DecreasingSplit, FixedSlice, DecreasingSlice

const CacheP = [Vector{IdWeight}(undef, 32)]

function __init__()
    while length(CacheP) < Threads.nthreads()
        push!(CacheP, copy(CacheP[1]))
    end
end

function reuse_cache!(n::Int)
    P = CacheP[Threads.threadid()]
    P = resize!(P, n)
    @inbounds for i in eachindex(P)
        P[i] = IdWeight(i, 0f0)
    end
    
    P    
end


abstract type SplitPolicy end

Base.@kwdef struct FixedSplit <: SplitPolicy
    nparts::Float32 = 3f0
end

Base.@kwdef struct DecreasingSplit <: SplitPolicy
    minsplit::Float32 = 2f0
    maxsplit::Int = 8
end

function next_split_size(div::FixedSplit, n::Int, iter::Int)
    ceil(Int, n / div.nparts)
end

next_split_size(div::DecreasingSplit, n::Int, iter::Int) = ceil(Int, n / max(div.minsplit, (div.maxsplit - (iter-1)^2)))

abstract type SlicePolicy end

Base.@kwdef struct FixedSlice <: SlicePolicy
    Δ::Float32 = 8
end

Base.@kwdef struct DecreasingSlice <: SlicePolicy
    minΔ::Int = 4
    maxΔ::Int = 12
end

next_vector_slice(slice::FixedSlice, dim::Int, iter::Int) = ceil(Int, log2(dim) * slice.Δ)
function next_vector_slice(slice::DecreasingSlice, dim::Int, iter::Int)
    #d = iter == 0 ? 0.0 : iter^1.5
    #d = iter == 0 ? 0.0 : (iter+1)^1.5
    #d = iter^1.7
    d = iter
    d = slice.maxΔ - d
    ceil(Int, log2(dim) * max(slice.minΔ, d))
end

struct SlicedEvaluation{DistType<:SemiMetric,DBType<:AbstractDatabase,SliceT<:SlicePolicy,SplitT<:SplitPolicy} <: AbstractSearchIndex
    dist::DistType
    db::DBType
    slice::SliceT
    split::SplitT
end

Base.copy(idx::SlicedEvaluation; dist=idx.dist, db=idx.db, slice=idx.slice, split=idx.split) = SlicedEvaluation(dist, db, slice, split)

function search_by_layers(idx::SlicedEvaluation, qvec, P_::Vector{IdWeight}, k::Integer)
    X = idx.db.matrix
    dist = idx.dist
    dim = size(X, 1)
    b = next_vector_slice(idx.slice, dim, 0)
    sp = 1
    ep = min(dim, sp + b - 1)
    P = view(P_, 1:length(P_))
    iter = 0

    @inbounds while sp <= dim
        iter += 1
        q = view(qvec, sp:ep)
        for (i, p) in enumerate(P)
            v = view(X, sp:ep, p.id)
            P[i] = IdWeight(p.id, p.weight + evaluate(dist, q, v))
        end

        n = next_split_size(idx.split, length(P), iter)
        if n > 8192
            partialsort!(P, n, by=p->p.weight)
        else
            partialsort!(P, 1:n, by=p->p.weight)
        end
        n <= k && break
        P = view(P_, 1:n)
        b = next_vector_slice(idx.slice, dim, iter)
        sp = ep + 1
        ep = min(dim, sp + b - 1)
    end
    
    P
end


function SimilaritySearch.searchbatch(idx::SlicedEvaluation, queries::AbstractDatabase, k::Integer; pools=nothing)
    n = length(idx.db)
    I = Matrix{Int32}(undef, k, n)
    D = Matrix{Float32}(undef, k, n)
    #GC.enable(false)
 
    Threads.@threads for i in eachindex(queries)
        P = search_by_layers(idx, queries[i], reuse_cache!(n), k)
        
        @inbounds for j in 1:k
            I[j, i] = P[j].id
        end
        
        @inbounds for j in 1:k
            D[j, i] = P[j].weight
        end
    end
    #GC.enable(true)
    
    I, D
end

function SimilaritySearch.search(idx::SlicedEvaluation, q, res::KnnResult; pools=nothing)
    P = search_by_layers(idx, queries[i], reuse_cache!(n), k)
    
    for i in 1:maxlength(resize)
        push!(res, P[i])
    end
    
    SearchResult(res, 0)
end

