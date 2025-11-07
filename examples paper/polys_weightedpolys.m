%% Section 5a Polynomials + weighted polynomials
% We consider the basis 
%       {p_i(x)}_{i=0}^N U {sqrt(x+1)p_i(x)}_{i=0}^N 
% for the approximation of 
%       f(x) = sqrt(x+1) / (1 + 5x^2) + cos(5x)
% on [-1,1].
addpath('../src'); clc; 

%% Illustration of computed "u" for fixed degree
rng(0);
N = 19; % number of basis functions = 2(N+1) = 40

% using rcs.m
disp('--- refinement-based Christoffel sampling (RCS) ---');
phi = @(x) evaluatebasis(x,N);
rhosampler = @(nbsamples) 2*rand(nbsamples,1) - 1;
weightedsampler = @(nbsamples, w) slicesample(rhosampler(1),nbsamples,"pdf",w);
indfun = @(x) (x >= -1).*(x <= 1);
integrator = @(f) mean(f(rhosampler(1000)));
maxchrist = 1e6;
rng(0); u0 = rcs(phi, rhosampler, weightedsampler, indfun, maxchrist, ...
    integrator, true, [], 0); 
rng(0); u4 = rcs(phi, rhosampler, weightedsampler, indfun, maxchrist, ...
    integrator, true, [], 4); 
rng(0); u8 = rcs(phi, rhosampler, weightedsampler, indfun, maxchrist, ...
    integrator, true, [], 8); 

% using the dense grid method
disp('--- dense grid method ---');
M = 10*maxchrist;
densegrid = rhosampler(M);
A = phi(densegrid)/sqrt(M);
[~,R] = qr([A; 1e-14*eye(size(A,2))], 0);
invchristoffel = @(x) (vecnorm(R'\phi(x)').^2)';

% comparison of u vs k_n^epsilon
xx = linspace(-1,1,500)';
f1 = figure('Position',[200 200 500 450]);
semilogy(xx,u0(xx),'-k'); hold on;
semilogy(xx,u4(xx),'-k', 'HandleVisibility','off');
semilogy(xx,u8(xx),'-k', 'HandleVisibility','off');
semilogy(xx,invchristoffel(xx),'--k');
ylim([1e1 10^(8)]); xlim([-1.1,1.1]);
xlabel('x'); figurestyle(20);
text(0,10^(6.25),'initialization','Interpreter','latex','FontSize',20,'HorizontalAlignment','center');
text(0,10^(3.9),'iteration 4','Interpreter','latex','FontSize',20,'HorizontalAlignment','center');
text(0,10^(1.9),'iteration 8','Interpreter','latex','FontSize',20,'HorizontalAlignment','center');
exportgraphics(f1, "weightedpolys_1.pdf", 'ContentType', 'vector');

dist = 2*logspace(-8,0,500)';
xx = -1 + dist;
f2 = figure('Position',[200 200 500 450]);
loglog(dist,u0(xx),'-k'); hold on;
loglog(dist,u4(xx),'-k', 'HandleVisibility','off');
loglog(dist,u8(xx),'-k', 'HandleVisibility','off');
loglog(dist,invchristoffel(xx),'--k');
xlim([-inf, 2]); ylim([1e1 10^8]);
xlabel('distance to the singularity at x = -1');
xticks([1e-7, 1e-5, 1e-3, 1e-1]);
figurestyle(20);  
text(10^(-2.5),10^(6.25),'initialization','Interpreter','latex','FontSize',20);
text(10^(-2.5),10^(5.3),'iteration 4','Interpreter','latex','FontSize',20,'Rotation',-35);
text(10^(-2.5),10^(3.5),'iteration 8','Interpreter','latex','FontSize',20,'Rotation',-38);
exportgraphics(f2, "weightedpolys_2.pdf", 'ContentType', 'vector');

%% Convergence test
Nlist = 3:6:75; 
repetitions = 10;
f = @(x) sqrt(x+1)./(1+5*x.^2) + cos(5*x);
errgrid = [2*logspace(-16,0,1000)'-1; cos((0:1000)*2*pi/1000)'];   
errf = f(errgrid);
errlist = zeros(length(Nlist),repetitions);
errlist_equi = zeros(length(Nlist),repetitions);

phi = @(x,n) evaluatebasis(x,n); 
rhosampler = @(nbsamples) 2*rand(nbsamples,1) - 1;
weightedsampler = @(nbsamples, w) slicesample(rhosampler(1),nbsamples,"pdf",w);
indfun = @(x) (x >= -1).*(x <= 1);
integrator = @(f) mean(f(rhosampler(1000)));
maxchrist = 1e6;
warning('off'); % ill-conditioned matrices
rng(0);

disp('--- convergence test ---');
for i = 1:length(Nlist)
    N = Nlist(i)
    errA = phi(errgrid,N);
    for j = 1:repetitions 
        % refinement-based Christoffel sampling
        [~, samplepoints, weights] = rcs(@(x) phi(x,N), rhosampler, ...
            weightedsampler, indfun, maxchrist, integrator, false); 
        A = sqrt(weights).*phi(samplepoints,N);
        F = sqrt(weights).*f(samplepoints);
        c = tsvd(A,F,1e-14);
        errlist(i,j) = max(abs(errf - errA*c));

        % uniformly random sampling (with the same number of sample points)
        samplepoints = rhosampler(length(samplepoints));
        weights = ones(length(samplepoints),1)/sqrt(length(samplepoints));
        A = sqrt(weights).*phi(samplepoints,N);
        F = sqrt(weights).*f(samplepoints);
        c = tsvd(A,F,1e-14);
        errlist_equi(i,j) = max(abs(errf - errA*c));
    end
end

% compute the geometric mean and variance
mean_curve = 10.^(mean(log10(errlist),2))';
std_curve = std(log10(errlist),0,2)';
curve_min = 10.^(log10(mean_curve) - std_curve);
curve_max = 10.^(log10(mean_curve) + std_curve);
mean_curve_equi = 10.^(mean(log10(errlist_equi),2))';
std_curve_equi = std(log10(errlist_equi),0,2)';
curve_min_equi = 10.^(log10(mean_curve_equi) - std_curve_equi);
curve_max_equi = 10.^(log10(mean_curve_equi) + std_curve_equi);

f3 = figure('Position',[200 200 500 450]);
semilogy(2*Nlist+1,mean_curve_equi,'^-k','MarkerSize',8, 'MarkerFaceColor',...
    'k','MarkerEdgeColor','k'); hold on;
fill([2*Nlist+1 fliplr(2*Nlist+1)], [curve_min_equi, fliplr(curve_max_equi)], ...
     'k', 'FaceAlpha', 0.08, 'EdgeColor', 'none', 'HandleVisibility', 'off');
semilogy(2*Nlist+1,mean_curve,'.-k','MarkerSize',30); 
fill([2*Nlist+1 fliplr(2*Nlist+1)], [curve_min, fliplr(curve_max)], ...
     'k', 'FaceAlpha', 0.08, 'EdgeColor', 'none', 'HandleVisibility', 'off');
xlabel('number of basis functions'); ylabel('uniform error');
figurestyle(20);
exportgraphics(f3, "weightedpolys_3.pdf", 'ContentType', 'vector');

%% Auxiliary functions
function phi = evaluatebasis(x,n)
    acosx = acos(x);
    px = cos((0:n).*acosx);
    phi = [px sqrt(1+x).*px];
end

function Z = arrayevaluation(f,x)
    z = f(x(:));
    Z = reshape(z,size(x,1),size(x,2));
end