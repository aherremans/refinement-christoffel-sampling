% Compute A\b using the truncated SVD with absolute tolerance atol.
function [x,r] = tsvd(A, b, atol)
    [U,S,V] = svd(A,0);
    s = diag(S);
    r = sum(s > atol);
    S_pinv = zeros(size(S))';
    S_pinv(1:r,1:r) = diag(s(1:r).^(-1));
    x = V*(S_pinv*(U'*b));
end