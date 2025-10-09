%% Section 5D Extreme learning machine (ELM)
% We consider the basis 
%       Phi_n = { g (a_i x + b_i) }_{i=1}^n
% for approximation on [0,1]^2, where g is a sigmoidal activation function,
% a_i is uniformly random in [-1,1]^2 and b_i is uniformly random in [0,1].
addpath('../'); clc; rng(0);

hiddennodes = 600;
elm = additiveELM(hiddennodes);
phi = @(x) evluate_additiveELM(x,elm);
rhosampler = @(nbsamples) rand(nbsamples,2);
weightedsampler = @(nbsamples, w) slicesample(rhosampler(1),nbsamples,"pdf",w);
indfun = @(x) (x(1) >= 0).*(x(1) <= 1).*(x(2) >= 0).*(x(2) <= 1);
integrator = @(f) mean(f(rhosampler(1000)));
maxchrist = 1e5;

tic; [u, samplepoints, weights] = rcs(phi, rhosampler, weightedsampler, ...
    indfun, maxchrist, integrator); toc;

%% plot section
x = linspace(0,1,100); y = linspace(0,1,100);
[X, Y] = meshgrid(x, y);

% plot u
uGrid = arrayevaluation(u,X,Y);
uGrid(uGrid == 0) = NaN;
f1 = figure; 
surf(X, Y, log10(uGrid), 'EdgeColor', 'none'); shading interp; 
xlabel('x'); ylabel('y'); zlabel('$\log_{10}(u)$');
figurestyle(22); colormap(parula(2048));
% exportgraphics(f1, "elmA.pdf", 'ContentType', 'vector');

% plot sample points (on top of u)
f2 = figure;
plot(samplepoints(:,1),samplepoints(:,2),'.k', 'MarkerSize',10);
xlabel('x'); ylabel('y');
figurestyle(22); 
% exportgraphics(f2, "elmB.pdf", 'ContentType', 'vector');

% plot some basis functions
phi_1 = arrayevaluation(@(x) evluate_additiveELM(x,elm,1), X, Y); 
phi_2 = arrayevaluation(@(x) evluate_additiveELM(x,elm,2), X, Y); 
f3 = figure; 
surf(X, Y, phi_1, 'EdgeColor', 'none'); shading interp;
xlabel('x'); ylabel('y'); zlabel('$\phi_1$');
figurestyle(22); 
% exportgraphics(f3, "elmC.pdf", 'ContentType', 'vector');
f4 = figure; 
surf(X, Y, phi_2, 'EdgeColor', 'none'); shading interp;
xlabel('x'); ylabel('y'); zlabel('$\phi_2$');
figurestyle(22); 
% exportgraphics(f4, "elmD.pdf", 'ContentType', 'vector');

%% Auxiliary functions
function elm = additiveELM(hiddennodes)
    elm = struct();
    elm.W = 2*rand(hiddennodes, 2) - 1;
    elm.b = rand(hiddennodes, 1);
end

function phi = evluate_additiveELM(x, elm, i)
    z = elm.W * x' + elm.b;
    a = 1./(1 + exp(-z));
    phi = a.'; 
    if(nargin > 2)
        phi = phi(:,i);
    end
end

function Z = arrayevaluation(f,x,y)
    z = f([x(:) y(:)]);
    Z = reshape(z,size(x,1),size(x,2));
end