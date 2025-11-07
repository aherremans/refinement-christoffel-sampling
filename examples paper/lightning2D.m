%% Section 5b Rational approximation with preassigned poles
% We consider the basis 
%       U_{j=1}^{n_1} {q_j p_k(x) p_l(y) / (Q(x,y) - q_j) }_{(k,l) = (1,1)}^(n_2,n_2}
%           U {p_k(x) p_l(y)}_{(k,l) = (1,1)}^(n_3,n_3}
% for approximation on [-2,2]^2.
%
% ! this script requires chebfun for plotting "u"
addpath('../src'); warning('off'); 

%% Illustration of "u" for fixed degree
rng(0);
n1 = 15; 
n2 = 3;        
n3 = 10;    
Q = @(x,y) x.^3 - 2*x + 1 - y.^2;
poles = [1i*exp(-4*(sqrt(n1) - sqrt(1:n1))) -1i*exp(-4*(sqrt(n1) - sqrt(1:n1)))];

phi = @(x) evaluatebasis(x(:,1),x(:,2),n1,n2,n3,Q);
indfun = @(x) (x(:,1) >= -2).*(x(:,1) <= 2).*(x(:,2) >= -2).*(x(:,2) <= 2);
rhosampler = @(numsamples) 4*rand(numsamples,2)-2;
weightedsampler = @(numsamples,pdf) slicesample(rhosampler(1),numsamples,"pdf",pdf,'Width',[2, 2]);
integrator = @(f) mean(f(rhosampler(1000)));
maxchrist = 100/min(abs(poles)); % this heuristic seemed to work well in the 1D case

[u, samplepoints, weights] = rcs(phi, rhosampler, weightedsampler, indfun, ...
   maxchrist, integrator);

%% plot section 
disp('(visualizing u...)');
% visualize u
f = chebfun2(Q, [-2 2 -2 2]); g = roots(f); % make sure the elliptic curve is well-represented in the plotting grid
curve1 = arrayfun(@(t) g(t,1), linspace(-1,1,25)');
curve2 = arrayfun(@(t) g(t,2), linspace(-1,1,25)');
x = [linspace(-2,2,50)'; real(curve1); real(curve2)];
y = [linspace(-2,2,50)'; imag(curve1); imag(curve2)];
x = sort(x); y = sort(y);
[X, Y] = meshgrid(x, y);
U = arrayevaluation(u,X,Y);
% I add some artifial smoothing to U, since otherwise the surf plot shows 
% big peaks due to the discretization (and these can be distracting). To 
% see that this is indeed justified, observe that u doesn't oscillate
% along the elliptic curve (Q(x,y) = 0):
%       plot3(real(curve1),imag(curve1),u([real(curve1) imag(curve1)])); hold on;
%       plot3(real(curve2),imag(curve2),u([real(curve2) imag(curve2)]));
%       set(gca,'Zscale','log'); zlim([1e3 1e7]);
U = imgaussfilt(U,1.5);
f1 = figure('Position',[200 200 500 450]);
surf(X, Y, log10(U), 'EdgeColor', 'none'); shading interp;
xlabel('x'); ylabel('y'); zlabel('$\log_{10}(u)$');
figurestyle(20); colormap(viridis);
% exportgraphics(f1, "lightning2D_1.pdf", 'ContentType', 'vector');

%% Convergence test
n1list = [3, 9, 12, 18, 21];
n2list = [2, 2, 3, 3, 3];
n3list = [2, 2, 5, 10, 15];
rng(0);

f = @(x) abs(x(:,1).^3 - 2*x(:,1) + 1 - x(:,2).^2);
[xc, yc] = compute_clustered_points(Q, 10, 30, [-2, 2, -2, 2]);
[xs, ys] = chebpts2(30, 30, [-2 2 -2 2]);
errgrid =[xc(:) yc(:); xs(:) ys(:)];
errf = f(errgrid);
repetitions = 10;
dofs = 2*n1list.*n2list.^2 + n3list.^2;
warning('off'); % ill-conditioned matrices

errlist = zeros(length(n1list),repetitions);
errlist_unif = zeros(length(n1list),repetitions);

tic; disp('--- convergence test ---');
for i = 1:length(n1list)
    n1 = n1list(i)
    n2 = n2list(i);
    n3 = n3list(i);
    phi = @(x) evaluatebasis(x(:,1),x(:,2),n1,n2,n3,Q);
    errA = phi(errgrid);
    poles = [1i*exp(-4*(sqrt(n1) - sqrt(1:n1))) -1i*exp(-4*(sqrt(n1) - sqrt(1:n1)))];
    maxchrist = 100/min(abs(poles));
    for j = 1:repetitions 
        disp(j);
        % refinement-based Christoffel sampling
        [~, samplepoints, weights] = rcs(@(x) phi(x), rhosampler, ...
            weightedsampler, indfun, maxchrist, integrator, false);
        m = size(samplepoints,1);
        A = sqrt(weights).*phi(samplepoints);
        F = sqrt(weights).*f(samplepoints);
        c = tsvd(A,F,1e-14);
        errlist(i,j) = max(abs(errf - errA*c));

        % uniformly random sampling (with the same number of sample points)
        samplepoints = rhosampler(m);
        weights = ones(m,1)/sqrt(m);
        A = sqrt(weights).*phi(samplepoints);
        F = sqrt(weights).*f(samplepoints);
        c = tsvd(A,F,1e-14);
        errlist_unif(i,j) = max(abs(errf - errA*c));
    end
end
toc;

%% Convergence plot
% compute the geometric mean and variance
mean_curve = 10.^(mean(log10(errlist),2))';
std_curve = std(log10(errlist),0,2)';
curve_min = 10.^(log10(mean_curve) - std_curve);
curve_max = 10.^(log10(mean_curve) + std_curve);
mean_curve_unif = 10.^(mean(log10(errlist_unif),2))';
std_curve_unif = std(log10(errlist_unif),0,2)';
curve_min_unif = 10.^(log10(mean_curve_unif) - std_curve_unif);
curve_max_unif = 10.^(log10(mean_curve_unif) + std_curve_unif);

f3 = figure('Position',[200 200 500 450]);
semilogy(dofs,mean_curve_unif,'^-k','MarkerSize',8, 'MarkerFaceColor', 'k'); hold on;
fill([dofs fliplr(dofs)], [curve_min_unif, fliplr(curve_max_unif)], ...
     'k', 'FaceAlpha', 0.08, 'EdgeColor', 'none', 'HandleVisibility', 'off');
semilogy(dofs,mean_curve,'.-k','MarkerSize',30); 
fill([dofs fliplr(dofs)], [curve_min, fliplr(curve_max)], ...
     'k', 'FaceAlpha', 0.08, 'EdgeColor', 'none', 'HandleVisibility', 'off');
xlabel('number of basis functions'); ylabel('uniform error');
figurestyle(20); ylim([1e-3, 1e16]); yticks([1e-3, 1e0, 1e5, 1e10, 1e15]);
% exportgraphics(f3, "lightning2D_2.pdf", 'ContentType', 'vector');

%% Auxiliary functions
function phi = evaluatebasis(x,y,n1,n2,n3,Q)
    n = length(x);
    acosx = acos(x/2);
    acosy = acos(y/2);
    px = cos((0:n2-1).*acosx);
    py = cos((0:n2-1).*acosy);
    sx = cos((0:n3-1).*acosx);
    sy = cos((0:n3-1).*acosy);
    pPoly = reshape(px, n, 1, n2).*reshape(py, n, n2, 1); 
    pPoly = reshape(pPoly, n, (n2)^2); 
    sPoly = reshape(sx, n, 1, n3).*reshape(sy, n, n3, 1);
    sPoly = reshape(sPoly, n, (n3)^2);
    denom = Q(x,y);
    poles = [1i*exp(-4*(sqrt(n1) - sqrt(1:n1))) -1i*exp(-4*(sqrt(n1) - sqrt(1:n1)))];
    scaling = 1./(denom - poles);
    rat = pPoly.*reshape(scaling, n, 1, 2*n1);
    rat = reshape(rat, n, []);
    phi = [rat sPoly];
end

function Z = arrayevaluation(f,x,y)
    z = f([x(:) y(:)]);
    Z = reshape(z,size(x,1),size(x,2));
end