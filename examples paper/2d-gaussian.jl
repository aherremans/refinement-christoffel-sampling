using FractionalFrames
using StaticArrays, SpecialFunctions, HypergeometricFunctions, LinearAlgebra
using Plots, LaTeXStrings, AbstractMCMC, SliceSampling, Statistics, Random
include("../src/rcs.jl")
include("../src/WeightModel.jl")

"""
Section 7.4:

Solving (I + (-Δ)^1/2) u(x,y) = f(x,y), where the solution is u(x,y) = exp(-x²-y²).
"""

# Gaussian Solution
function ua(xy)
    x, y = first(xy), last(xy)
    exp(-x^2-y^2)
end

# RHS corresponding to (-Δ)^(1/2) exp(-x^2-y^2)
function ga(xy, s)
    x, y = first(xy), last(xy)
    4^s * gamma(s+1) *_₁F₁(s+1, 1, -x^2-y^2)
end

Random.seed!(0)

T = Float64
s = 1/2;
T̃ = ExtendedZernike(0.0, -s)
W = ExtendedWeightedZernike(0.0, s)

L = AbsLaplacianPower(axes(W,1), s)
P = L*T̃
Q = L*W

# These are going to be the outer radii of the disks we consider
js = [1, 3/2, 2, 3, 4]

# Sum space, interlacing degree, functions, and translations.
Sp = [];S_ = [];
for j in js
    append!(Sp, [SumSpace{Float64, Tuple{typeof(T̃), typeof(W)}}((T̃,W), [-j, j])])
    append!(S_, [SumSpace{Float64, Tuple{typeof((1/j).*P),typeof((1/j).*Q)}}(((1/j).*P,(1/j).*Q), [-j, j])])
end
Sₚ = SumSpace{Float64, NTuple{length(Sp), eltype(Sp)}}(Tuple(Sp), [-1.,1.])
S = SumSpace{Float64, NTuple{length(S_), eltype(S_)}}(Tuple(S_), [-1.,1.])

# Can consider two kind of collocation points:
#
# (1) pick radial collocation points (avoiding edges of disks where things blow up)
#     then tensor those with angular collocation points "collocation_points_disk".

# (2) Square tensor product of collocation points "collocation_points_square"

function collocation_points_disk(M, Me)
    r = collocation_points(M, Me, I=vcat(0, js), endpoints=[eps(),10*one(T)], innergap=1e-3)
    r = r[Me+1:end]

    θ = range(0, 2π, 5) 
    SVector.(r.*cos.(θ)', r.*sin.(θ)')
end

function collocation_points_square()
    SVector.(range(0,10,470), range(0,10,470)')
end

# Since solution is radially symmetric, I only want
# to evaluate columns with Fourier mode (0,0)
mode_0(n) = 1 + sum(1:2*n)
function zero_mode_columns(N::Int, perN::Int)
    a = mode_0.(0:N÷perN-1) .* perN
    b = [a[i]-perN+1:a[i] for i in 1:length(a)]
    vcat(b...)
end

# RCS sampler
rhosampler = (n) -> begin
    θ = 2π * rand(n,1)           
    r = 10 * sqrt.(rand(n,1))   
    hcat(r.*cos.(θ), r.*sin.(θ)) 
end
indfun(x) = x[:,1].^2 + x[:,2].^2 .<= 10. ^2
maxchrist = 30000
integrator(f) = mean(f(rhosampler(1000)))
function weightedsampler(n, w)
    model = AbstractMCMC.LogDensityModel(WeightModel.WeightedLogDensityModel(w,2))
    chain = sample(model, RandPermGibbs(SliceSteppingOut(10.)), n; initial_params=vec(rand(2,1)))
    return (vcat([s.params' for s in chain]...), [exp(s.lp[1]) for s in chain])
end

#### We will use an independent error grid
xy_err = SVector{2, Float64}.(eachrow(rhosampler(5000)))
UA = ua.(xy_err)
ufun = nothing
Nlist = 10:10:110
reps = 10

#### Solve
errors = zeros(Float64,length(Nlist))
nrm_cfs = zeros(Float64,length(Nlist))
nb_samples = zeros(Float64,length(Nlist))

errors_rcs = zeros(Float64,length(Nlist),reps)
nrm_cfs_rcs = zeros(Float64,length(Nlist),reps)
nb_samples_rcs = zeros(Float64,length(Nlist),reps)

errors_unif = zeros(Float64,length(Nlist),reps)
nrm_cfs_unif = zeros(Float64,length(Nlist),reps)

for i = 1:length(Nlist)
    N = Nlist[i]
    cols = zero_mode_columns(N, 10) # extract zero-Fourier mode columns
    Aₚ = Sₚ[xy_err, cols]       # Assemble least-square matrix for frame of solution
    
    ### Hard-coded sampling
    xy = collocation_points_disk(N,N) # scale collocation points linearly
    A = S[xy[:], cols]     # Assemble least-squares matrix for frame of RHS
    u = A[:,1:N] \ ga.(xy[:], s)
    err = abs.(Aₚ[:,1:N]*u-UA)  # Inf-norm error at independent error grid
    errors[i] =  norm(err, Inf)
    nrm_cfs[i] = norm(u, Inf)
    nb_samples[i] = length(xy)

    for j = 1:reps
        ### RCS sampling (C3 = 10)
        phi(xy) = S[SVector{2, Float64}.(eachrow(xy)), cols]
        ufun, xy, w = rcs(phi, rhosampler, weightedsampler, indfun, maxchrist, integrator; C3=5)
        xy = SVector{2, Float64}.(eachrow(xy))
        A = S[xy, cols]     # Assemble least-squares matrix for frame of RHS
        u = (sqrt.(w) .* A[:, 1:N]) \ (sqrt.(w) .* ga.(xy, s))
        err = abs.(Aₚ[:,1:N]*u-UA)  # Inf-norm error at independent error grid
        errors_rcs[i,j] = norm(err, Inf)
        nrm_cfs_rcs[i,j] = norm(u, Inf)
        nb_samples_rcs[i,j] = length(xy)

        ### Uniform sampling (using the same number of points as RCS)
        xy = SVector{2, Float64}.(eachrow(rhosampler(length(xy))))
        A = S[xy, cols]     # Assemble least-squares matrix for frame of RHS
        u = A[:,1:N] \ ga.(xy, s)
        err = abs.(Aₚ[:,1:N]*u-UA)  # Inf-norm error at independent error grid
        errors_unif[i,j] = norm(err, Inf)
        nrm_cfs_unif[i,j] = norm(u, Inf)
    end
    
    print("Completed n = $N.\n")
end

# Compute mean and std curves (log scale)
mean_curve = 10 .^ mean(log10.(errors_rcs), dims=2)
std_curve = std(log10.(errors_rcs), dims=2)
curve_min = 10 .^ (log10.(mean_curve) .- std_curve)
curve_max = 10 .^ (log10.(mean_curve) .+ std_curve)
mean_curve_unif = 10 .^ mean(log10.(errors_unif), dims=2)
std_curve_unif = std(log10.(errors_unif), dims=2)
curve_min_unif = 10 .^ (log10.(mean_curve_unif) .- std_curve_unif)
curve_max_unif = 10 .^ (log10.(mean_curve_unif) .+ std_curve_unif)
p = plot(Nlist, errors,
    linewidth=2,
    markershape=:xcross,
    markersize=5,
    ylabel="uniform error",
    xlabel="number of basis functions",
    yscale=:log10,
    xtickfontsize=12, ytickfontsize=12,xlabelfontsize=15,ylabelfontsize=15,
    gridlinewidth = 2,
    legend=:none,
    yticks=[1e-9,1e-6,1e-3,1e0,1e3],
    xticks=0:20:100,
    ylims=(1e-9,1e5),
    color=:black,
    label="", 
    size=(500,450),
    fontfamily="Computer Modern",
    ygridlinewidth=0.8,
    xgridlinewidth=0.8,
    framestyle=:box
)
plot!(
    Nlist, mean_curve_unif, 
    color=:black, 
    yscale=:log10,
    linewidth=2,
    marker = :utriangle,                
    markersize = 6,
    label=""
)
plot!(Nlist, mean_curve, 
    marker=:circle, 
    color=:black, 
    markersize=6,
    linewidth=2,
    label=""
)
plot!(vcat(Nlist, reverse(Nlist)),
      vcat(curve_max, reverse(curve_min)),
      seriestype=:shape,
      color=:black,
      alpha=0.08,
      label="",
      linewidth=0)
plot!(vcat(Nlist, reverse(Nlist)),
      vcat(curve_max_unif, reverse(curve_min_unif)),
      seriestype=:shape,
      color=:black,
      alpha=0.08,
      label="",
      linewidth=0)
Plots.savefig("2Dgaussian_1.pdf")


# plot of the number of samples
mean_curve_samples = mean(nb_samples_rcs, dims=2)
std_curve_samples = std(nb_samples_rcs, dims=2, corrected=false)
p = plot(Nlist, nb_samples,
    linewidth=2,
    markershape=:xcross,
    markersize=5,
    ylabel="number of samples",
    xlabel="number of basis functions",
    xtickfontsize=12, ytickfontsize=12,xlabelfontsize=15,ylabelfontsize=15,
    gridlinewidth = 2,
    legend=:none,
    xticks=0:20:100,
    color=:black,
    size=(500,450),
    fontfamily="Computer Modern",
    ygridlinewidth=0.8,
    xgridlinewidth=0.8,
    framestyle=:box
)
plot!(Nlist, mean_curve_samples, 
    marker=:circle, 
    color=:black, 
    markersize=6,
    linewidth=2
)
plot!(Nlist, mean_curve_samples, 
    ribbon=std_curve_samples,
    fillalpha=0.08, color=:black, label="", linewidth=0
)
Plots.savefig("2Dgaussian_2.pdf")

# Plot the inverse Christoffel function
r_max = 10.
n = 200
theta = range(0, 2π, length=n)
r = range(0, r_max, length=n)
X = [ρ*cos(φ) for ρ in r, φ in theta]
Y = [ρ*sin(φ) for ρ in r, φ in theta]
Z = [ufun([x y])[1] for (x, y) in zip(X, Y)]
surface(
    X, Y, log10.(Z), 
    xlabel=L"$x$", 
    ylabel=L"$y$", 
    zlabel=L"$\log_{10}(u)$", 
    c=:viridis, 
    cbar=nothing, 
    size=(500,450), 
    margin=(-10., :mm),
    xtickfontsize=10, 
    ytickfontsize=10, 
    ztickfontsize=10, 
    xlabelfontsize=12, 
    ylabelfontsize=12, 
    zlabelfontsize=12,
    fontfamily="Computer Modern",
    ygridlinewidth=0.8,
    xgridlinewidth=0.8
)
yticks!([-10,-5,0,5,10])
xticks!([-10,-5,0,5,10])
Plots.savefig("2Dgaussian_3.pdf")