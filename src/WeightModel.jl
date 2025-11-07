module WeightModel

using LogDensityProblems

"""
A LogDensityProblems.jl-compatible model for slice sampling
based on an arbitrary weight function w(x).
"""
struct WeightedLogDensityModel
    w::Function     # weight function (1 x d -> 1-vector)
    d::Int          # dimension
end

# LogDensityProblems interface
LogDensityProblems.capabilities(::Type{WeightedLogDensityModel}) = LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.dimension(::Type{WeightedLogDensityModel}) = model.d

function LogDensityProblems.logdensity(model::WeightedLogDensityModel, x::AbstractArray)
    X = reshape(x, 1, model.d)
    v = model.w(X)[1]
    return (v > 0 && isfinite(v)) ? log(v) : -Inf
end

end