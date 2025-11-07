%% Section 5c Fourier extension of a triangular curve
% We consider the basis 
%       {exp(2*pi*i*(kx*x + ky*y))}_{(kx,ky) = (-n1, -n1)}^{(n1, n1)}
% for approximation on the edge of triangle ⊂ [-1/2 1/2]^2.
addpath('../src'); clc;

%% visualizing u for a fixed degree
rng(0);
n = 15; % number of basis functions = (2*n+1)^2
A = [0; 0.4]; 
B = [0.3; -0.45];
C = [-0.4; -0.1];

phi = @(t) evaluatebasis(t,n,A,B,C);
indfun = @(t) (t >= 0).*(t <= 1);
rhosampler = @(nbsamples) rand(nbsamples,1);
weightedsampler = @(nbsamples,pdf) slicesample(rand,nbsamples,"pdf",pdf);
integrator = @(f) mean(f(rand(1000,1)));
maxchrist = 7000;

disp('--- refinement-based Christoffel sampling (RCS) ---');
[u, samplepoints, weights] = rcs(phi, rhosampler, weightedsampler, indfun, ...
    maxchrist, integrator,[],10*n+50); 

disp('--- dense grid method ---');
M = 40*maxchrist;
densegrid = rhosampler(M);
T = phi(densegrid)/sqrt(M);
[~,R] = qr([T; 1e-14*eye(size(T,2))], 0);
invchristoffel = @(x) (vecnorm(R'\phi(x)').^2)';

%% visualize u
tt = linspace(0,1,1000)';
cart = paramtocartesian(A,B,C,tt);
xx = cart(:,1); yy = cart(:,2);

f1 = figure('Position',[200 200 500 450]);
plot3(xx,yy,u(tt),'k'); hold on;
xlim([-1/2,1/2]); ylim([-1/2,1/2]);
plot3(xx,yy,invchristoffel(tt),'--k');
set(gca,'ZScale','log'); zticks([1e2, 1e3]);
zlabel("$\log_{10}(u)$");
view(-20,30);
figurestyle(20);
% exportgraphics(f1, "triangle_1.pdf");

%% show convergence
rng(0);
f = @(x) sqrt(x(:,1).^2 + x(:,2).^2);
nlist = 3:3:18;
tt = linspace(0,1,20000)';
errgrid = paramtocartesian(A,B,C,tt);
errf = f(errgrid);
repetitions = 10;
errlist = zeros(length(nlist),repetitions);
errlist_equi = zeros(length(nlist),repetitions);
phi = @(t,n) evaluatebasis(t,n,A,B,C); 
indfun = @(t) (t >= 0).*(t <= 1);
rhosampler = @(nbsamples) rand(nbsamples,1);
weightedsampler = @(nbsamples,pdf) slicesample(rand,nbsamples,"pdf",pdf);
integrator = @(f) mean(f(rand(1000,1)));
maxchrist = 5000;
numericaldimension = @(n) min((2*n+1)^2, 10*n+50);

warning('off'); % ill-conditioned matrices
disp('--- convergence test ---');

for i = 1:length(nlist)
    n = nlist(i); 
    disp('number of basis functions :' + string((2*n+1)^2)); 
    disp('estimate of the numerical dimension :' + string(numericaldimension(n)));
    errT = phi(tt,n);
    for j = 1:repetitions 
        disp('rep :'+ string(j));
        % refinement-based Christoffel sampling
        [u, samplepoints, weights] = rcs(@(x) phi(x,n), ...
            rhosampler, weightedsampler, indfun, maxchrist, integrator, true, numericaldimension(n)); 
        cartpoints = paramtocartesian(A,B,C,samplepoints);
        T = sqrt(weights).*phi(samplepoints,n);
        F = sqrt(weights).*f(cartpoints);
        c = tsvd(T,F,1e-14);
        errlist(i,j) = max(abs(errf - errT*c));

        % uniformly random sampling (with the same number of sample points)
        samplepoints = rhosampler(length(samplepoints));
        weights = ones(length(samplepoints),1)/sqrt(length(samplepoints));
        cartpoints = paramtocartesian(A,B,C,samplepoints);
        T = sqrt(weights).*phi(samplepoints,n);
        F = sqrt(weights).*f(cartpoints);
        c = tsvd(T,F,1e-14);
        errlist_equi(i,j) = max(abs(errf - errT*c));
    end
end

%% plot section
% compute the geometric mean and variance
mean_curve = 10.^(mean(log10(errlist),2))';
std_curve = std(log10(errlist),0,2)';
curve_min = 10.^(log10(mean_curve) - std_curve);
curve_max = 10.^(log10(mean_curve) + std_curve);
mean_curve_equi = 10.^(mean(log10(errlist_equi),2))';
std_curve_equi = std(log10(errlist_equi),0,2)';
curve_min_equi = 10.^(log10(mean_curve_equi) - std_curve_equi);
curve_max_equi = 10.^(log10(mean_curve_equi) + std_curve_equi);

nbfuns = (2*nlist+1).^2;
f2 = figure('Position',[200 200 500 450]);
semilogy(nbfuns,mean_curve_equi,'^-k','MarkerSize',8,'MarkerFaceColor','k'); hold on;
fill([nbfuns fliplr(nbfuns)], [curve_min_equi, fliplr(curve_max_equi)], ...
     'k', 'FaceAlpha', 0.08, 'EdgeColor', 'none', 'HandleVisibility', 'off');
semilogy(nbfuns,mean_curve,'.-k','MarkerSize',25); 
fill([nbfuns fliplr(nbfuns)], [curve_min, fliplr(curve_max)], ...
     'k', 'FaceAlpha', 0.08, 'EdgeColor', 'none', 'HandleVisibility', 'off');
xlabel('number of basis functions'); ylabel('uniform error');
figurestyle(20); 
% exportgraphics(f2, "triangle_2.pdf", 'ContentType', 'vector');

%% auxiliary function
function x = paramtocartesian(A,B,C,t)
    L1 = norm(B - A);
    L2 = norm(C - B); 
    L3 = norm(A - C); 
    t = (L1 + L2 + L3)*t;
    x = zeros(length(t),2);
    for i = 1:length(t)
        if (t(i) < L1)
            s = t(i)/L1;
            x(i, :) = (1 - s)*A + s*B;
        elseif (t(i) < L1 + L2)
            s = (t(i) - L1) / L2;
            x(i, :) = (1 - s)*B + s*C;
        else
            s = (t(i)-L1-L2)/L3;
            x(i, :) = (1 - s)*C + s*A;
        end
    end
end

function phi = evaluatebasis(t,n,A,B,C)
    cart = paramtocartesian(A,B,C,t);
    x = cart(:,1); y = cart(:,2);
    [xfreq, yfreq] = meshgrid(1:n, 1:n);
    xfreq = xfreq(:)'; yfreq = yfreq(:)';
    sinx = sin(2*pi*x*xfreq); cosx = cos(2*pi*x*xfreq);
    siny = sin(2*pi*y*yfreq); cosy = cos(2*pi*y*yfreq);
    phi = [ones(length(x),1), sinx.*siny, sinx.*cosy, cosx.*siny, cosx.*cosy, ...
        sin(2*pi*x*(1:n)), cos(2*pi*x*(1:n)), sin(2*pi*y*(1:n)), cos(2*pi*y*(1:n))];
end
