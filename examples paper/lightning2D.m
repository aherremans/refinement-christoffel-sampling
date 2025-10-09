%% Section 5B Rational approximation with preassigned poles
% We consider the basis 
%       U_{j=1}^{n_1} {q_j p_k(x) p_l(y) / (Q(x,y) - q_j) }_{(k,l) = (1,1)}^(n_2,n_2}
%           U {p_k(x) p_l(y)}_{(k,l) = (1,1)}^(n_3,n_3}
% for approximation on [-2,2]^2.
%
% ! this script requires chebfun for plotting "u"
addpath('../'); clc; rng(0); warning('off');

n1 = 15; 
n2 = 3;        
n3 = 10;    
Q = @(x,y) x.^3 - 2*x + 1 - y.^2;
poles = [1i*exp(-4*(sqrt(n1) - sqrt(1:n1))) -1i*exp(-4*(sqrt(n1) - sqrt(1:n1)))];

phi = @(x) evaluatebasis(x(:,1),x(:,2),n1,n2,n3,Q);
indfun = @(x) (x(:,1) >= -2).*(x(:,1) <= 2).*(x(:,2) >= -2).*(x(:,2) <= 2);
rhosampler = @(numsamples) 4*rand(numsamples,2)-2;
weightedsampler = @(numsamples,pdf) slicesample(rhosampler(1),numsamples,"pdf",pdf);
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
% heavily along the elliptic curve (Q(x,y) = 0):
%       plot3(real(curve1),imag(curve1),u([real(curve1) imag(curve1)])); hold on;
%       plot3(real(curve2),imag(curve2),u([real(curve2) imag(curve2)]));
%       set(gca,'Zscale','log'); zlim([1e3 1e7]);
U = imgaussfilt(U,1.5);
f1 = figure; 
surf(X, Y, log10(U), 'EdgeColor', 'none'); shading interp;
xlabel('x'); ylabel('y'); zlabel('$\log_{10}(u)$');
figurestyle(24);
% exportgraphics(f1, "lightning2DA.pdf", 'ContentType', 'vector');

% visualize samplepoints 
f2 = figure;
plot(samplepoints(:,1),samplepoints(:,2),'.k','MarkerSize',10);
xlabel('x'); ylabel('y');
figurestyle(24);
% exportgraphics(f2, "lightning2DB.pdf", 'ContentType', 'vector');

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