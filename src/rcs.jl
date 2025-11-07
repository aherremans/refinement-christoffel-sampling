using LinearAlgebra

"""
    rcs(phi, rhosampler, weightedsampler, indfun, maxchrist; integrator, 
        verbose=true, numericaldim=nothing, maxiter=50, parallelise=false)

Refinement-based Christoffel sampling (RCS).

# Arguments
- phi(x::AbstractMatrix) -> matrix: evaluate basis at rows of x (returns nX x n).
- rhosampler(num::Integer) -> matrix: draws num samples (rows).
- weightedsampler(num::Integer, u::Function) -> (matrix, matrix): draws num weighted samples given weight function u and evaluates u(samplepoints).
- indfun(x::AbstractMatrix) -> Vector{Bool} mask for rows of x (length nX).
- maxchrist::Real scalar upper bound to the maximum of the inverse Christoffel function.
- integrator(f::Function) -> Real: estimates ∫_X f dρ where f(samples) returns a vector of sample values.

# Keyword arguments
- verbose::Bool=true
- numericaldim::Int=nothing
- maxiter::Int=50

# Returns
- u : function accepting x (rows = points) returning u(x).
- samplepoints : matrix of output samples (rows = points).
- weights : vector of sample weights.

"""
function rcs(phi, rhosampler, weightedsampler, indfun, maxchrist, integrator; 
    verbose::Bool=true, numericaldim::Union{Nothing, Int}=nothing, maxiter::Int=50, C3::Int=10)
    
    # numeric constants
    EPS = 1e-14
    C1 = 5
    C2 = 5*C1

    uinit = x -> (indfun(x) .* float(maxchrist))     # returns vector length = nX

    s1 = rhosampler(1)
    if numericaldim === nothing
        n = size(phi(s1), 2)
    else
        n = numericaldim
    end
    d = size(s1, 2)

    RList = Vector{LowerTriangular{Float64, Matrix{Float64}}}()  # store lower-triangular matrices (R')
    numsamples = round(Int, C2 * n)
    l1norm = integrator(x -> uinit(x))
    converged = false
    iter = 1
    samplepoints = Array{Float64,2}(undef, 0, d)
    weights = Float64[]

    while (!converged) && (iter <= maxiter)
        # Assess convergence by checking whether alpha <= 1
        alpha = (C2 / C1) * n / l1norm
        if alpha >= 1
            numsamples = round(Int, C1 * l1norm)
            converged = true
        end

        # Construct A
        if iter == 1
            samplepoints = rhosampler(numsamples)
            weights = ones(numsamples) ./ (maxchrist * C1)
        else
            u = x -> evalU(phi, x, indfun, RList, uinit, d)
            (samplepoints, uvec) = weightedsampler(numsamples, u)
            weights = 1.0 ./ (uvec * C1)
        end

        # A = sqrt(weights) .* phi(samplepoints)   (weights per row)
        Φ = phi(samplepoints)                       # nX x n
        sqrtw = sqrt.(weights)
        A = Φ .* sqrtw[:, ones(Int, size(Φ,2))]     # broadcast rows

        # Update u by precomputing QR decomposition of augmented A
        # Augment with EPS * norm(A,2) * I
        aug = vcat(A, EPS * opnorm(A, 2) * Matrix(I, n, n))
        F = qr!(aug)                                 # thin QR; F.R is upper triangular (n x n)
        R = F.R
        # store R' (lower triangular) to match MATLAB logic
        L = LowerTriangular(R')                     # store as lower triangular matrix
        push!(RList, L)
        
        if length(RList) == 3
            # keep only last two entries
            RList = RList[end-1:end]
        end

        # Estimate L1 norm of u
        l1norm = integrator(samples -> evalU(phi, samples, indfun, RList, uinit, d))

        if verbose
            @info "Estimate of ∫_X u dρ after iteration $(iter): $(l1norm)"
        end

        iter += 1
    end

    if !converged
        @warn "rcs did not converge within the maximum number of iterations."
        u = x -> evalU(phi, x, indfun, RList, uinit, d)
        samplepoints = Array{Float64,2}(undef, 0, d)
        weights = Float64[]
        return u, samplepoints, weights
    else
        # Sample the final output set
        numsamples = round(Int, C3 * l1norm)
        u = x -> evalU(phi, x, indfun, RList, uinit, d)
        (samplepoints, uvec) = weightedsampler(numsamples, u)
        weights = 1.0 ./ (uvec .* C3)
        return u, samplepoints, weights
    end
end


"""
    evalU(phi, x, indfun, RList, uinit, d)

Evaluate the u function at rows of x. Returns a vector with one value per row.
"""
function evalU(phi, x, indfun, RList, uinit, d::Int)
    if size(x,2) != d
        throw(ArgumentError("evalU: wrong x format; expected $d columns, got $(size(x,2))"))
    end

    Δ = 1.75
    nR = length(RList)
    T = phi(x)                   
    nX = size(x,1)
    mask = indfun(x)           
    uVals = uinit(x)            

    norms = zeros(nX, max(0, nR))
    Tt = transpose(T)
    for (j, Rlower) in enumerate(RList)
        Y = Rlower \ Tt
        col_norms_sq = vec([sum(abs2, view(Y, :, k)) for k = 1:size(Y,2)])
        norms[:, j] = Δ * col_norms_sq
    end
    vals = hcat(uVals, norms)
    vals[.!mask, :] .= 0
    out = mapslices(minimum, vals; dims=2)

    return vec(out)
end
