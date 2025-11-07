using LinearAlgebra

"""
    tsvd(A, b, atol)

Compute the least-squares solution x of A*x ≈ b using the truncated SVD,
keeping only singular values greater than atol. Returns x and the rank r.
"""
function tsvd(A, b, atol)
    F = svd(A)  # full SVD
    U, S, V = F.U, F.S, F.V
    r = sum(S .> atol)  # number of singular values above atol
    S_pinv = Diagonal(vcat(1 ./ S[1:r], zeros(length(S) - r)))  # pseudo-inverse
    x = V * S_pinv * (U' * b)
    return x
end