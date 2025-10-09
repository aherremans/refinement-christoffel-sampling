%% Section 5B Rational approximation with preassigned poles
% We consider the basis 
%       {-q_i / (x - q_i) }_{i=1}^{n_1} U {p_i(x)}_{i=1}^{n_2}
% for approximation on [0,1].
addpath('../'); clc; rng(0);

%% Illustration of "u" for fixed degree
rhosampler = @(nbsamples) rand(nbsamples,1);
weightedsampler = @(nbsamples, w) slicesample(rhosampler(1),nbsamples,"pdf",w);
indfun = @(x) (x >= 0).*(x <= 1);
integrator = @(f) integral(@(x) arrayevaluation(f,x),0,1,'RelTol',1e-2);

% n1 = 4, n2 = 1
disp('--- refinement-based Christoffel sampling (RCS) ---');
n1 = 4; n2 = 1;
poles = -exp(4*(sqrt(1:1:n1) - sqrt(n1)));
phi = @(x) [x.^(0:n2-1), -poles./(x-poles)];
maxchrist = 1e4;
u1 = rcs(phi, rhosampler, weightedsampler, indfun, maxchrist, integrator);
disp('--- dense grid method ---');
M = 40*maxchrist;
densegrid = rhosampler(M);
A = phi(densegrid)/sqrt(M);
[~,R] = qr([A; 1e-14*eye(size(A,2))], 0);
invchristoffel1 = @(x) (vecnorm(R'\phi(x)').^2)';

% n1 = 8, n2 = 3
disp('--- refinement-based Christoffel sampling (RCS) ---');
n1 = 8; n2 = 3;
poles = -exp(4*(sqrt(1:1:n1) - sqrt(n1)));
phi = @(x) [x.^(0:n2-1), -poles./(x-poles)];
maxchrist = 1e5;
u2 = rcs(phi, rhosampler, weightedsampler, indfun, maxchrist, integrator);
disp('--- dense grid method ---');
M = 40*maxchrist;
densegrid = rhosampler(M);
A = phi(densegrid)/sqrt(M);
[~,R] = qr([A; 1e-14*eye(size(A,2))], 0);
invchristoffel2 = @(x) (vecnorm(R'\phi(x)').^2)';

% n1 = 12, n2 = 5
disp('--- refinement-based Christoffel sampling (RCS) ---');
n1 = 12; n2 = 5;
poles = -exp(4*(sqrt(1:1:n1) - sqrt(n1)));
phi = @(x) [x.^(0:n2-1), -poles./(x-poles)];
maxchrist = 1e6;
u3 = rcs(phi, rhosampler, weightedsampler, indfun, maxchrist, integrator);
disp('--- dense grid method ---');
M = 40*maxchrist;
densegrid = rhosampler(M);
A = phi(densegrid)/sqrt(M);
[~,R] = qr([A; 1e-14*eye(size(A,2))], 0);
invchristoffel3 = @(x) (vecnorm(R'\phi(x)').^2)';

% plot
xx = logspace(-10,0,1000)';
f1 = figure;
loglog(xx,u1(xx),'k'); hold on;
loglog(xx,invchristoffel1(xx),'--k');
loglog(xx,u2(xx),'k'); 
loglog(xx,invchristoffel2(xx),'--k');
loglog(xx,u3(xx),'k'); 
loglog(xx,invchristoffel3(xx),'--k');
legend('$u$ (RCS)','$k_n^\epsilon$ (dense grid method)', ...
    'Location','southwest');
xlabel('x');
ylim([1e0,1e6]); xlim([1e-10,1]);
figurestyle(22); 
pos = f1.Position; pos(4) = 520; f1.Position = pos;   
text(1e-9,10^(5.53),'$n_1 = 12, n_2 = 5$','Interpreter','latex','FontSize',20);
text(1e-9,10^(4.38),'$n_1 = 8, n_2 = 3$','Interpreter','latex','FontSize',20);
text(1e-9,10^(2.98),'$n_1 = 4, n_2 = 1$','Interpreter','latex','FontSize',20);
% exportgraphics(f1, "lightningA.pdf", 'ContentType', 'vector');

%% Convergence test
n1list = 2:6:62;
errgrid = [0; logspace(-16,0,1000)'];
f = @(x) sqrt(x);
errf = f(errgrid);
reps = 10;
errlist = zeros(length(n1list),reps);
errlist_equi = zeros(length(n1list),reps);
warning('off'); % ill-conditioned matrices
rhosampler = @(nbsamples) rand(nbsamples,1);
weightedsampler = @(nbsamples, w) slicesample(rhosampler(1),nbsamples,"pdf",w); 
indfun = @(x) (x >= 0).*(x <= 1);
integrator = @(f) integral(@(x) arrayevaluation(f,x),0,1,'RelTol',1e-2);

for i = 1:length(n1list)
    n1 = n1list(i)
    n2 = round(2*sqrt(n1));
    poles = -exp(4*(sqrt(1:1:n1) - sqrt(n1)));   
    phi = @(x) [x.^(0:n2-1), -poles./(x-poles)];   
    maxchrist = 100/min(abs(poles));  % this seems to work pretty well
    errA = phi(errgrid);
    for j = 1:reps
        % refinement-based Christoffel sampling
        [u, samplepoints, weights] = rcs(phi, rhosampler, weightedsampler, ...
            indfun, maxchrist, integrator, false);
        A = sqrt(weights).*phi(samplepoints);
        F = sqrt(weights).*f(samplepoints);
        c = tsvd(A,F,1e-14);
        errlist(i,j) = max(abs(errf - errA*c));

        % uniformly random sampling
        samplepoints = rhosampler(length(samplepoints));
        weights = ones(length(samplepoints),1)/sqrt(length(samplepoints));
        A = sqrt(weights).*phi(samplepoints);
        F = sqrt(weights).*f(samplepoints);
        c = tsvd(A,F,1e-14);
        errlist_equi(i,j) = max(abs(errf - errA*c));
    end
end

mean_curve = 10.^(mean(log10(errlist),2))';
std_curve = std(log10(errlist),0,2)';
curve_min = 10.^(log10(mean_curve) - std_curve);
curve_max = 10.^(log10(mean_curve) + std_curve);
mean_curve_equi = 10.^(mean(log10(errlist_equi),2))';
std_curve_equi = std(log10(errlist_equi),0,2)';
curve_min_equi = 10.^(log10(mean_curve_equi) - std_curve_equi);
curve_max_equi = 10.^(log10(mean_curve_equi) + std_curve_equi);

f2 = figure;
semilogy(n1list,mean_curve_equi,'d-k','MarkerSize',10); hold on;
fill([n1list fliplr(n1list)], [curve_min_equi, fliplr(curve_max_equi)], ...
     'k', 'FaceAlpha', 0.08, 'EdgeColor', 'none', 'HandleVisibility','off');
semilogy(n1list,mean_curve,'.-k','MarkerSize',25); 
fill([n1list fliplr(n1list)], [curve_min, fliplr(curve_max)], ...
     'k', 'FaceAlpha', 0.08, 'EdgeColor', 'none');
xlabel('$n_1$ (number of clustered poles)');
ylabel('uniform error');
legend('uniformly random', 'RCS','Location','southwest');
figurestyle(22); xlim([-inf,n1list(end)]);
pos = f2.Position; pos(4) = 520; f2.Position = pos;
ylim([1e-8,1e2]);
% exportgraphics(f2, "lightningB.pdf", 'ContentType', 'vector');

%% Auxiliary functions 
function Z = arrayevaluation(f,x)
    z = f(x(:));
    Z = reshape(z,size(x,1),size(x,2));
end