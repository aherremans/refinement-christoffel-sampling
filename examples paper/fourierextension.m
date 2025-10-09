%% Section 5C Fourier extension of a surface
% We consider the basis 
%       {exp(2*pi*i*(kx*x + ky*y + kz*z))}_{(kx,ky,kz) = (-n1, -n1, -n1)}^{(n1, n1, n1)}
% for approximation on the surface of cube ⊂ [-1/2 1/2]^3.
addpath('../'); clc; rng(0);

n = 4;      % number of basis functions = (2n + 1)^3
L = 0.2;

% ! we now call christoffelsampling.m with x = [u v] --> two parameters
%   which parametrize the surface of the cube
phi = @(x) evaluatebasis(x(:,1),x(:,2),n,L);
indfun = @(x) (x(:,1) >= 0).*(x(:,1) <= 1).*(x(:,2) >= 0).*(x(:,2) <= 1);
rhosampler = @(nbsamples) rand(nbsamples,2);
weightedsampler = @(nbsamples,pdf) slicesample(rhosampler(1),nbsamples,"pdf",pdf);
integrator = @(f) mean(f(rhosampler(1000)));
maxchrist = 5e4;

[u, samplepoints, weights] = rcs(phi, rhosampler, weightedsampler, indfun, ...
    maxchrist, integrator);

%% plot section
% visualize u 
faces = [0, 1/6, 1/3, 1/2, 2/3, 5/6, 1];
f1 = figure; hold on
for k = 1:length(faces)-1 
    uRange = linspace(faces(k)+eps, faces(k+1)-eps, 40);
    vRange = linspace(0,1,40);
    [U,V] = meshgrid(uRange, vRange);
    [X,Y,Z] = paramtocartesian(U(:),V(:),L);
    X = reshape(X,size(U));
    Y = reshape(Y,size(U));
    Z = reshape(Z,size(U));
    U = reshape(u([U(:) V(:)]), size(U));
    surf(X,Y,Z,log10(U),'EdgeColor','none')
end
axis equal
view([-37.5 10]); xlim([-0.35,0.35]); ylim([-0.35,0.35]); zlim([-0.35,0.35]);
c = colorbar; c.Ticks = [2.5, 3, 3.5, 4, 4.5];
c.TickLabels = {'$10^2$', '$10^{2.5}$', '$10^3$', '$10^{3.5}$', '$10^4$', '$10^{4.5}$'};
xlabel('x'); ylabel('y'); zlabel('z'); 
figurestyle(22); 
% exportgraphics(f1, "fourierA.pdf", 'ContentType', 'vector');

% visualize sample points
[xs,ys,zs] = paramtocartesian(samplepoints(:,1),samplepoints(:,2),L);
f2 = figure;
scatter3(xs, ys, zs, 3, 'filled','MarkerFaceColor','k');
axis equal
view([-37.5 10]); xlim([-0.35,0.35]); ylim([-0.35,0.35]); zlim([-0.35,0.35]);
xlabel('x'); ylabel('y'); zlabel('z'); 
figurestyle(22);
% exportgraphics(f2, "fourierB.pdf", 'ContentType', 'vector');


%% Auxiliary function
function [x,y,z] = paramtocartesian(u,v,L)
    % the surface is a cube with side length 2L and rotated such that it
    % doesn't align with the axes
    %   0 <= u <= 1, 0 <= v <= 1
    pts = zeros(length(u),3);
    % first sample the unit cube
    for i = 1:length(u)
        if(u(i) <= 1/6)
            pts(i,1) = 1;
            pts(i,2) = 6*u(i);
            pts(i,3) = v(i);
        elseif(u(i) <= 1/3)
            pts(i,1) = 1 - 6*(u(i) - 1/6);
            pts(i,2) = 1;
            pts(i,3) = v(i);
        elseif(u(i) <= 1/2)
            pts(i,1) = 0;
            pts(i,2) = 1 - 6*(u(i) - 1/3);
            pts(i,3) = v(i);
        elseif(u(i) <= 2/3)
            pts(i,1) = 6*(u(i) - 1/2);
            pts(i,2) = 0;
            pts(i,3) = v(i);
        elseif(u(i) <= 5/6)
            pts(i,1) = 6*(u(i) - 2/3);
            pts(i,2) = v(i);
            pts(i,3) = 1;
        elseif(u(i) <= 1)
            pts(i,1) = 6*(u(i) - 5/6);
            pts(i,2) = v(i);
            pts(i,3) = 0;
        end
    end
    % scale + translate + rotate the cube
    pts = pts - 0.5*ones(length(u),3);
    pts = 2*L*pts;
    Rx = [1 0 0;
      0 cos(pi/4) -sin(pi/4);
      0 sin(pi/4) cos(pi/4)];
    Ry = [cos(pi/4) 0 sin(pi/4);
          0 1 0;
          -sin(pi/4) 0 cos(pi/4)];
    R = Ry * Rx; 
    pts = (R*pts')';
    x = pts(:,1); y = pts(:,2); z = pts(:,3);
end

% This is significantly faster than using complex exponentials
function phi = evaluatebasis(u,v,n,L)
    [x,y,z] = paramtocartesian(u,v,L);
    % 1D terms
    terms1D = [sin(2*pi*x*(1:n)), cos(2*pi*x*(1:n)), sin(2*pi*y*(1:n)), cos(2*pi*y*(1:n)), ...
        sin(2*pi*z*(1:n)), cos(2*pi*z*(1:n))];
    % 2D terms
    [freq1, freq2] = meshgrid(1:n, 1:n); freq1 = freq1(:)'; freq2 = freq2(:)';
    sinx = sin(2*pi*x*freq1); cosx = cos(2*pi*x*freq1);
    siny = sin(2*pi*y*freq2); cosy = cos(2*pi*y*freq2);
    termsXY = [sinx.*siny, sinx.*cosy, cosx.*siny, cosx.*cosy];
    sinz = sin(2*pi*z*freq2); cosz = cos(2*pi*z*freq2);
    termsXZ = [sinx.*sinz, sinx.*cosz, cosx.*sinz, cosx.*cosz];
    siny = sin(2*pi*y*freq1); cosy = cos(2*pi*y*freq1);
    termsYZ = [siny.*sinz, siny.*cosz, cosy.*sinz, cosy.*cosz];
    % 3D terms
    [xfreq,yfreq,zfreq] = ndgrid(1:n,1:n,1:n);
    xfreq = xfreq(:)'; yfreq = yfreq(:)'; zfreq = zfreq(:)';
    sinx = sin(2*pi*x*xfreq); cosx = cos(2*pi*x*xfreq);
    siny = sin(2*pi*y*yfreq); cosy = cos(2*pi*y*yfreq);
    sinz = sin(2*pi*z*zfreq); cosz = cos(2*pi*z*zfreq);
    termsXYZ = [ sinx.*siny.*sinz, sinx.*siny.*cosz, sinx.*cosy.*sinz, ...
        sinx.*cosy.*cosz, cosx.*siny.*sinz, cosx.*siny.*cosz, ...
        cosx.*cosy.*sinz, cosx.*cosy.*cosz];

    phi = [ones(length(x),1), terms1D, termsXY, termsXZ, termsYZ, termsXYZ];
end