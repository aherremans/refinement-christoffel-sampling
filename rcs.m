function [u,samplepoints, weights] = rcs(phi, rhosampler, ...
    weightedsampler, indfun, maxchrist, integrator, verbose, numericaldim, maxiter)
    % The refinement-based Christoffel sampling (RCS) algorithm.
    %
    % INPUT:
    %   - phi: Function handle that evaluates the basis.
    %          phi(x) = [phi_1(x) ... phi_n(x)]
    %          where x is a matrix with each row representing a point in 
    %          the domain X.
    %
    %   - rhosampler: Function handle that draws samples according to rho.
    %                 rhoSampler(numSamples) = matrix of i.i.d. samples
    %                 from the measure rho; each row is a point in X.
    %
    %   - weightedsampler: Function handle that draws samples according to a weighted measure.
    %                      weightedSampler(numSamples, w) = matrix of i.i.d. 
    %                      samples from mu, where dmu ~ w drho (! w is not
    %                      normalized); each row is a point in X.
    %
    %   - indfun: Function handle representing the indicator function on the domain X.
    %             indFun(x) = logical array indicating whether each row of 
    %             x is in X.
    %
    %   - maxchrist: Scalar upper bound on the maximum value of the inverse 
    %                (numerical) Christoffel function over the domain X.
    %
    %   - integrator: Function handle that estimates integration over X.
    %                 integrator(f) ≈ int_X f drho
    %                 where f is a function handle accepting a matrix x
    %                 with rows in X.
    %
    % OPTIONAL INPUT:
    %   - verbose: boolean indicating whether progress should be printed
    %   - numericaldim: (overestimate of) the numerical dimension
    %   - maxiter: maximum number of iterations
    %
    if (nargin < 6)
        throw("Not enough input arguments given.");
    end
    if (nargin < 7 || isempty(verbose))
        verbose = true;
    end
    if (nargin < 8 || isempty(numericaldim))
        n = size(phi(rhosampler(1)), 2);
    else
        n = numericaldim;
    end
    if (nargin < 9 || isempty(maxiter))
        maxiter = 50;
    end

    EPS = 1e-14;                        % Regularization parameter
    C1 = 5;       
    C2 = 5*C1;    
    C3 = 10;

    RList = {};                         % Stores QR decompositions for evaluation of u
    numsamples = round(C2*n);       
    uinit = @(x) indfun(x)*maxchrist;
    l1norm = integrator(@(samples) uinit(samples));
    d = length(rhosampler(1));
    converged = false;
    i = 1;

    while (~converged && i <= maxiter)
        % Assess convergence by checking whether alpha <= 1
        alpha = (C2/C1)*n/l1norm;
        if (alpha >= 1)                 % perform one more step with alpha = 1
            numsamples = round(C1*l1norm);
            converged = true;
        end

        % Construct A
        if (i == 1)
            samplepoints = rhosampler(numsamples);      
            weights = ones(numsamples,1)/(maxchrist*C1);
        else
            u = @(x) evalU(phi, x, indfun, RList, uinit, d);
            samplepoints = weightedsampler(numsamples, u);
            weights = 1./(u(samplepoints)*C1);
        end
        A = sqrt(weights).*phi(samplepoints);

        % Update u (by precomputing a QR decomposition)
        [~, R] = qr([A; EPS*norm(A)*eye(size(A, 2))], 0);
        RList{i} = R;
        
        % Estimate the L^1 norm of u
        l1norm = integrator(@(samples) evalU(phi, samples, indfun, RList, uinit, d));
        if(verbose)
            disp(strcat('Estimate of int_X u drho after iteration -',string(i),'- : ',string(l1norm)));
        end

        i = i + 1;
    end

    if (~converged)
        disp('christoffelsampling.m did not converge within the maximum number of iterations.');
        u = @(x) evalU(phi, x, indfun, RList, uinit, d);
        samplepoints = [];
        weights = [];
    else
        % Sample the output sample set and construct B
        numsamples = round(C3*l1norm);
        u = @(x) evalU(phi, x, indfun, RList, uinit, d);
        samplepoints = weightedsampler(numsamples, u);
        weights = 1./(u(samplepoints)*C3);
        B = sqrt(weights).*phi(samplepoints)/sqrt(1-3/4);
        u = @(x) evalU(phi, x, indfun, RList, uinit, d);
    end
end

% Auxiliary function to evaluate u; accepts matrices x with each row
% representing a point in the domain X.
function val = evalU(phi, x, indFun, RList, uInit, d)
    if(size(x,2) ~= d)
        disp(size(x));
        throw('Error in evalU: wrong "x" format.')
    end
    K = 2;
    list = zeros(min(length(RList),K)+1, 1);
    T = phi(x);    
    val = zeros(size(x,1),1);
    for i = 1:size(x,1)
        if (~indFun(x(i,:)))
            val(i) = 0;
        else
            v = T(i,:)';
            list(1) = uInit(x(i,:));
            for j = 1:min(length(RList),K)
                R = RList{end-j+1};
                list(j+1) = (1+3/4)*norm(R'\v)^2;
            end
            val(i) = min(list);
        end
    end
end