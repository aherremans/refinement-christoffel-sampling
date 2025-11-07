using Random, LinearAlgebra, Distributions, Plots, SliceSampling, LogDensityProblems, AbstractMCMC, LaTeXStrings
include("../src/rcs.jl") 
include("../src/WeightModel.jl")
include("../src/tsvd.jl")


function weightedsampler(n, w)
    model = AbstractMCMC.LogDensityModel(WeightModel.WeightedLogDensityModel(w,1))
    chain = sample(model, SliceSteppingOut(.1), n; initial_params=[rand(Float64)])
    return ([c.params[1] for c in chain], [exp(c.lp[1]) for c in chain])
end

function make_phi(n1, n2)
    poles = -exp.(4 .* (sqrt.(1:n1) .- sqrt(n1)))
    return x -> hcat([x.^k for k in 0:(n2-1)]..., [-p ./ (x .- p) for p in poles]...)
end

function dense_inv_christoffel(phi, M, rhosampler)
    densegrid = rhosampler(M)
    A = phi(densegrid) ./ sqrt(M)
    F = qr(vcat(A, 1e-14 * Matrix(I, size(A, 2), size(A, 2))))
    R = F.R
    return x -> norm(R' \ phi(x)')^2
end


Random.seed!(0)

rhosampler(n) = rand(n, 1)
indfun(x) = (x .>= 0) .& (x .<= 1)
integrator(f) = mean(f(rand(1000,1))[:])

n1list = 2:6:68
errgrid = vcat(0., 10 .^ range(-16,0,1000))
f(x) = sqrt(x)
errf = f.(errgrid)
reps = 10

errlist = zeros(length(n1list),reps)
errlist_DGM1 = zeros(length(n1list)-3,reps)
errlist_DGM2 = zeros(length(n1list)-3,reps)
timing = zeros(length(n1list),reps)
timing_DGM1 = zeros(length(n1list)-3,reps)
timing_DGM2 = zeros(length(n1list)-3,reps)

for i = 1:length(n1list)
    n1 = n1list[i]
    @show n1
    n2 = round(Int,2sqrt(n1))
    poles = -exp.(4(sqrt.(1:n1) .- sqrt(n1)))
    phi = make_phi(n1,n2)
    maxchrist = 100. ./ minimum(abs.(poles))
    errA = phi(errgrid)
    for j = 1:reps 
        @show(j)
        # Refinement-based Christoffel sampling
        timing[i,j] = @elapsed begin 
            u, samplepoints, weights = rcs(phi, rhosampler, weightedsampler, indfun, maxchrist, integrator; verbose=false, C3=15)
        end
        A = sqrt.(weights) .* phi(samplepoints)
        F = sqrt.(weights) .* f.(samplepoints)
        c = tsvd(A,F,1e-14)
        errlist[i,j] = maximum(abs.(errf - errA*c))
        m = length(samplepoints)
        
        if i <= length(n1list) - 3
            # Dense grid method using 1e4 points
            M = 10000
            timing_DGM1[i,j] = @elapsed begin ic = dense_inv_christoffel(phi, M, rhosampler)
            icfun = x -> (indfun(x)*ic(x))
            (samplepoints, icvec) = weightedsampler(m,icfun)
            weights = integrator(icfun) ./ (icvec * m)
            end
            A = sqrt.(weights) .* phi(samplepoints)
            F = sqrt.(weights) .* f.(samplepoints)
            c = tsvd(A,F,1e-14)
            errlist_DGM1[i,j] = maximum(abs.(errf - errA*c))

            # Dense grid method using 1e7 points
            M = 10000000
            timing_DGM2[i,j] = @elapsed begin ic = dense_inv_christoffel(phi, M, rhosampler)
            icfun = x -> (indfun(x)*ic(x))
            (samplepoints, icvec) = weightedsampler(m,icfun)
            weights = integrator(icfun) ./ (icvec * m)
            end
            A = sqrt.(weights) .* phi(samplepoints)
            F = sqrt.(weights) .* f.(samplepoints)
            c = tsvd(A,F,1e-14)
            errlist_DGM2[i,j] = maximum(abs.(errf - errA*c))
        end
    end
end

# number of basis functions 
nbfuns = n1list + round.(Int,2sqrt.(n1list))
nbfuns_DGM = nbfuns[1:end-3]

# error plots
mean_curve      = 10 .^ mean(log10.(errlist), dims=2)
std_curve       = std(log10.(errlist), dims=2)
curve_min       = 10 .^ (log10.(mean_curve) .- std_curve)
curve_max       = 10 .^ (log10.(mean_curve) .+ std_curve)
mean_curve_DGM1  = 10 .^ mean(log10.(errlist_DGM1), dims=2)
std_curve_DGM1   = std(log10.(errlist_DGM1), dims=2)
curve_min_DGM1   = 10 .^ (log10.(mean_curve_DGM1) .- std_curve_DGM1)
curve_max_DGM1   = 10 .^ (log10.(mean_curve_DGM1) .+ std_curve_DGM1)
mean_curve_DGM2  = 10 .^ mean(log10.(errlist_DGM2), dims=2)
std_curve_DGM2   = std(log10.(errlist_DGM2), dims=2)
curve_min_DGM2   = 10 .^ (log10.(mean_curve_DGM2) .- std_curve_DGM2)
curve_max_DGM2   = 10 .^ (log10.(mean_curve_DGM2) .+ std_curve_DGM2)

plt = plot(
    nbfuns, mean_curve, 
    markershape=:circle, 
    color=:black, 
    ms=5, 
    yscale=:log10, 
    linewidth=1,
    label="",
    size=(500,450),
    fontfamily="Computer Modern",
    ygridlinewidth=0.8,
    xgridlinewidth=0.8,
    framestyle=:box,
    xtickfontsize=12, ytickfontsize=12,xlabelfontsize=15,ylabelfontsize=15
)
plot!(nbfuns_DGM, mean_curve_DGM1, markershape=:square, color=:black, ms=5, linewidth=1, label="")
plot!(nbfuns_DGM, mean_curve_DGM2, markershape=:square, color=:black, ms=5, linewidth=1, label="")
plot!(vcat(nbfuns, reverse(nbfuns)), vcat(curve_min, reverse(curve_max)), seriestype =:shape, color=:black, alpha=0.08, label="", linewidth=0)
plot!(vcat(nbfuns_DGM, reverse(nbfuns_DGM)), vcat(curve_min_DGM1, reverse(curve_max_DGM1)), seriestype=:shape, color=:black, alpha=0.08, label="", linewidth=0)
plot!(vcat(nbfuns_DGM, reverse(nbfuns_DGM)), vcat(curve_min_DGM2, reverse(curve_max_DGM2)), seriestype=:shape, color=:black, alpha=0.08, label="", linewidth=0)
xlabel!("number of basis functions")
ylabel!("uniform error")
annotate!(73, 0.05, text(L"\ell = 10^4", 12))
annotate!(73, 0.0002, text(L"\ell = 10^7", 12))
# Plots.savefig("lightning_timing_1.pdf")

# error plots
mean_curve_timing     = mean(timing, dims=2)
std_curve_timing       = std(timing, dims=2)
curve_min_timing       = mean_curve_timing .- std_curve_timing
curve_max_timing       = mean_curve_timing .+ std_curve_timing
mean_curve_DGM1_timing  = mean(timing_DGM1, dims=2)
std_curve_DGM1_timing   = std(timing_DGM1, dims=2)
curve_min_DGM1_timing   = mean_curve_DGM1_timing .- std_curve_DGM1_timing
curve_max_DGM1_timing   = mean_curve_DGM1_timing .+ std_curve_DGM1_timing
mean_curve_DGM2_timing  = mean(timing_DGM2, dims=2)
std_curve_DGM2_timing   = std(timing_DGM2, dims=2)
curve_min_DGM2_timing   = mean_curve_DGM2_timing .- std_curve_DGM2_timing
curve_max_DGM2_timing   = mean_curve_DGM2_timing .+ std_curve_DGM2_timing
plt2 = plot(
    nbfuns, mean_curve_timing, 
    markershape=:circle, 
    color=:black, 
    ms=5, 
    linewidth=1, 
    label="",
    size=(500,450),
    fontfamily="Computer Modern",
    ygridlinewidth=0.8,
    xgridlinewidth=0.8,
    framestyle=:box,
    xtickfontsize=12, ytickfontsize=12,xlabelfontsize=15,ylabelfontsize=15,
    ylims=(-8,Inf)
)
plot!(nbfuns_DGM, mean_curve_DGM1_timing, markershape=:square, color=:black, ms=5, linewidth=1, label="")
plot!(nbfuns_DGM, mean_curve_DGM2_timing, markershape=:square, color=:black, ms=5, linewidth=1, label="")
plot!(vcat(nbfuns, reverse(nbfuns)), vcat(curve_min_timing, reverse(curve_max_timing)), seriestype =:shape, color=:black, alpha=0.08, label="", linewidth=0)
plot!(vcat(nbfuns_DGM, reverse(nbfuns_DGM)), vcat(curve_min_DGM1_timing, reverse(curve_max_DGM1_timing)), seriestype=:shape, color=:black, alpha=0.08, label="", linewidth=0)
plot!(vcat(nbfuns_DGM, reverse(nbfuns_DGM)), vcat(curve_min_DGM2_timing, reverse(curve_max_DGM2_timing)), seriestype=:shape, color=:black, alpha=0.08, label="", linewidth=0)
xlabel!("number of basis functions")
ylabel!("elapsed time (sec)")
annotate!(55, -4, text(L"\ell = 10^4", 12))
annotate!(47,28, text(L"\ell = 10^7", 12))
# Plots.savefig("lightning_timing_2.pdf")