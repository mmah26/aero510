% AE510 Project - Problem 3 (CFRP orthotropic panel)
% Condensed MATLAB version in lecture/HW8 style (SI units).
clear; clc;

%% Geometry and base load (SI)
IN_TO_M = 0.0254;
L = 56.0 * IN_TO_M;                  % m
W = 4.0  * IN_TO_M;                  % m
t = 0.5  * IN_TO_M;                  % m

p_ref = 1e6;                         % Pa
tmag = p_ref * t;                    % N/m line traction

% fixed mesh (~100 elements)
[xy, conn] = gen_quad_mesh(L, W, 20, 5);

%% CFRP constituent properties
Ef = 241e9;  nuf = 0.2;
Em = 3.12e9; num = 0.38;
Vf = 0.60;

[E1, E2, G12, nu12] = cfrp_ud_effective(Ef, nuf, Em, num, Vf);
fprintf('Problem 3 CFRP effective properties:\n');
fprintf('  E1  = %.6e Pa\n', E1);
fprintf('  E2  = %.6e Pa\n', E2);
fprintf('  G12 = %.6e Pa\n', G12);
fprintf('  nu12= %.6f\n\n', nu12);

%% Output folder
this_file = mfilename('fullpath');
[this_dir,~,~] = fileparts(this_file);
out_dir = fullfile(this_dir, 'figures', 'panel_composite', 'matlab');
if exist(out_dir, 'dir') ~= 7
    mkdir(out_dir);
end

%% Load cases (traction direction on right edge)
case_tags = {'case1','case2','case3','case4','case5'};
case_angles = [0, 30, 45, 60, 90]; % deg

% baseline outputs at theta=0
theta_ref = 0.0;
baseline_rows = [];
for icase = 1:5
    tvec = tmag * [cosd(case_angles(icase)); sind(case_angles(icase))];
    bc = dirichlet_bcs_for_edge(xy, 'left', [1 2], 0.0);
    % Baseline solve at theta=0 to report direct comparables across cases.
    out = solve_quad_case_ortho(xy, conn, t, E1, E2, G12, nu12, theta_ref, ...
                                struct('edge','right','tvec',tvec), bc);
    max_u = max(out.u_mag);
    max_abs_sx = max(abs(out.sig_elem(:,1)));
    max_abs_sy = max(abs(out.sig_elem(:,2)));
    max_vm = max(out.vm_elem);
    baseline_rows = [baseline_rows; icase, max_u, max_abs_sx, max_abs_sy, max_vm]; %#ok<AGROW>

    fprintf('Case %d @ theta=0 deg: max_u=%.4e m, max|sx|=%.4e Pa, max|sy|=%.4e Pa, max_vm=%.4e Pa\n', ...
        icase, max_u, max_abs_sx, max_abs_sy, max_vm);
end
fprintf('\n');

%% Fiber orientation sweep
angles = (0:5:90).';
nang = length(angles);
vm_case = zeros(nang,5);
sx_case = zeros(nang,5);
sy_case = zeros(nang,5);

for ia = 1:nang
    theta = angles(ia);
    for icase = 1:5
        tvec = tmag * [cosd(case_angles(icase)); sind(case_angles(icase))];
        bc = dirichlet_bcs_for_edge(xy, 'left', [1 2], 0.0);
        % Sweep orthotropic panel response versus fiber angle.
        out = solve_quad_case_ortho(xy, conn, t, E1, E2, G12, nu12, theta, ...
                                    struct('edge','right','tvec',tvec), bc);
        sx_case(ia,icase) = max(abs(out.sig_elem(:,1)));
        sy_case(ia,icase) = max(abs(out.sig_elem(:,2)));
        vm_case(ia,icase) = max(out.vm_elem);
    end
end

% Recommendations
[vm1_min, i1] = min(vm_case(:,1));
best_case1_theta = angles(i1);

worst_arb = max(vm_case(:,2:5), [], 2); % minimax metric for arbitrary loading set
[vm_arb_min, iarb] = min(worst_arb);
best_arb_theta = angles(iarb);

fprintf('Recommended fiber orientation:\n');
fprintf('  Case 1 (+x traction): theta = %.1f deg, min max_vm = %.6e Pa\n', best_case1_theta, vm1_min);
fprintf('  Cases 2-5 (minimax):  theta = %.1f deg, worst max_vm = %.6e Pa\n', best_arb_theta, vm_arb_min);

%% Write outputs
write_csv(fullfile(out_dir,'problem3_baseline_theta0.csv'), ...
    'case_id,max_u_m,max_abs_sigma_x_pa,max_abs_sigma_y_pa,max_sigma_vm_pa', baseline_rows);

ang_table = [angles, sx_case, sy_case, vm_case, worst_arb];
head = ['theta_deg,' ...
        'sx_case1_pa,sx_case2_pa,sx_case3_pa,sx_case4_pa,sx_case5_pa,' ...
        'sy_case1_pa,sy_case2_pa,sy_case3_pa,sy_case4_pa,sy_case5_pa,' ...
        'vm_case1_pa,vm_case2_pa,vm_case3_pa,vm_case4_pa,vm_case5_pa,' ...
        'worst_vm_cases2to5_pa'];
write_csv(fullfile(out_dir,'problem3_orientation_sweep.csv'), head, ang_table);

fid = fopen(fullfile(out_dir,'problem3_summary.txt'),'w');
fprintf(fid,'E1_pa=%.8e\n',E1);
fprintf(fid,'E2_pa=%.8e\n',E2);
fprintf(fid,'G12_pa=%.8e\n',G12);
fprintf(fid,'nu12=%.8e\n',nu12);
fprintf(fid,'best_case1_theta_deg=%.1f\n',best_case1_theta);
fprintf(fid,'best_case1_max_vm_pa=%.8e\n',vm1_min);
fprintf(fid,'best_arbitrary_theta_deg=%.1f\n',best_arb_theta);
fprintf(fid,'best_arbitrary_worst_vm_pa=%.8e\n',vm_arb_min);
fclose(fid);

%% Local functions
function [E1, E2, G12, nu12] = cfrp_ud_effective(Ef, nuf, Em, num, Vf)
% UD lamina effective constants from rule of mixtures + Halpin-Tsai.
Vm = 1.0 - Vf;
Gf = Ef/(2*(1+nuf));
Gm = Em/(2*(1+num));
E1 = Vf*Ef + Vm*Em;
E2 = halpin_tsai(Ef, Em, Vf, 2.0);
G12 = halpin_tsai(Gf, Gm, Vf, 1.0);
nu12 = Vf*nuf + Vm*num;
end

function P = halpin_tsai(Pf, Pm, Vf, xi)
% Compact Halpin-Tsai relation used for E2 and G12 estimates.
r = Pf/Pm;
eta = (r-1)/(r+xi);
P = Pm * (1 + xi*eta*Vf) / (1 - eta*Vf);
end

function out = solve_quad_case_ortho(xy, conn, t, E1, E2, G12, nu12, theta_deg, loads, bc)
Nn = size(xy,1); Ne = size(conn,1); dof = 2;
ndof = Nn*dof;
[gps, w] = gauss2x2();
D = Qbar_plane_stress(E1, E2, G12, nu12, theta_deg);

I = zeros(Ne*64,1); J = I; V = I; nz = 0;
F = zeros(ndof,1);

for e = 1:Ne
    nodes = conn(e,:);
    xy_e = xy(nodes,:);
    Ke = zeros(8,8); fe = zeros(8,1);

    for ig = 1:4
        xi = gps(ig,1); eta = gps(ig,2);
        [N, dN_dxi, dN_deta] = shape_quad4(xi, eta);
        [~, detJ, dN_dxdy] = jacobian_quad4(xy_e, dN_dxi, dN_deta);
        B = Bmat_quad4(dN_dxdy);
        Ke = Ke + (B' * D * B) * detJ * t * w(ig);
    end

    if element_on_edge(xy, xy_e, loads.edge)
        fe = fe + traction_edge_vector(xy_e, loads.edge, loads.tvec);
    end

    % Element-to-global dof map: [u1 v1 u2 v2 u3 v3 u4 v4].
    edofs = zeros(8,1);
    for a = 1:4
        edofs(2*a-1) = 2*nodes(a)-1;
        edofs(2*a) = 2*nodes(a);
    end

    F(edofs) = F(edofs) + fe;
    for ii = 1:8
        for jj = 1:8
            nz = nz + 1;
            I(nz) = edofs(ii);
            J(nz) = edofs(jj);
            V(nz) = Ke(ii,jj);
        end
    end
end
K = sparse(I(1:nz), J(1:nz), V(1:nz), ndof, ndof); % global sparse stiffness

c_dofs = (bc(:,1)-1)*dof + bc(:,2);
c_vals = bc(:,3);
[c_dofs, ia] = unique(c_dofs, 'stable');
c_vals = c_vals(ia);
all_dofs = (1:ndof).';
free_dofs = setdiff(all_dofs, c_dofs);
u = zeros(ndof,1); u(c_dofs) = c_vals;
% Static condensation solve on free dofs.
u(free_dofs) = K(free_dofs,free_dofs) \ (F(free_dofs) - K(free_dofs,c_dofs)*u(c_dofs));

sig_elem = zeros(Ne,3);
vm_elem = zeros(Ne,1);
for e = 1:Ne
    nodes = conn(e,:);
    xy_e = xy(nodes,:);
    ue = zeros(8,1);
    for a = 1:4
        ue(2*a-1) = u(2*nodes(a)-1);
        ue(2*a) = u(2*nodes(a));
    end
    [~, dN_dxi, dN_deta] = shape_quad4(0,0); % centroid stress sample
    [~, ~, dN_dxdy] = jacobian_quad4(xy_e, dN_dxi, dN_deta);
    B = Bmat_quad4(dN_dxdy);
    sig = D * (B * ue);
    sig_elem(e,:) = sig.';
    sx = sig(1); sy = sig(2); txy = sig(3);
    vm_elem(e) = sqrt(sx^2 - sx*sy + sy^2 + 3*txy^2);
end

u_nodes = reshape(u,2,[]).';
out.u_nodes = u_nodes;
out.u_mag = sqrt(sum(u_nodes.^2,2));
out.sig_elem = sig_elem;
out.vm_elem = vm_elem;
end

function D = Qbar_plane_stress(E1, E2, G12, nu12, theta_deg)
% Build transformed in-plane orthotropic matrix in global x-y axes.
nu21 = nu12 * E2 / E1;
den = 1 - nu12*nu21;
Q11 = E1/den; Q22 = E2/den; Q12 = nu12*E2/den; Q66 = G12;

m = cosd(theta_deg); n = sind(theta_deg);
m2 = m*m; n2 = n*n; m4 = m2*m2; n4 = n2*n2;

Qb11 = Q11*m4 + 2*(Q12+2*Q66)*m2*n2 + Q22*n4;
Qb22 = Q11*n4 + 2*(Q12+2*Q66)*m2*n2 + Q22*m4;
Qb12 = (Q11+Q22-4*Q66)*m2*n2 + Q12*(m4+n4);
Qb16 = (Q11-Q12-2*Q66)*m*m2*n - (Q22-Q12-2*Q66)*m*n2*n;
Qb26 = (Q11-Q12-2*Q66)*m*n2*n - (Q22-Q12-2*Q66)*m*m2*n;
Qb66 = (Q11+Q22-2*Q12-2*Q66)*m2*n2 + Q66*(m4+n4);

D = [Qb11 Qb12 Qb16; Qb12 Qb22 Qb26; Qb16 Qb26 Qb66];
end

function bc = dirichlet_bcs_for_edge(xy, edge, dof_ids, val)
ids = edge_node_ids(xy, edge);
bc = zeros(length(ids)*length(dof_ids), 3);
k = 0;
for i = 1:length(ids)
    for j = 1:length(dof_ids)
        k = k + 1;
        bc(k,:) = [ids(i), dof_ids(j), val];
    end
end
end

function ids = edge_node_ids(xy, edge)
x = xy(:,1); y = xy(:,2);
span = max(xy,[],1) - min(xy,[],1);
tol = max(max(span)*1e-10, 1e-12);
switch edge
    case 'left',   ids = find(abs(x - min(x)) <= tol);
    case 'right',  ids = find(abs(x - max(x)) <= tol);
    case 'bottom', ids = find(abs(y - min(y)) <= tol);
    case 'top',    ids = find(abs(y - max(y)) <= tol);
    otherwise, error('Unsupported edge.');
end
end

function on = element_on_edge(xy, xy_e, edge)
xmin = min(xy(:,1)); xmax = max(xy(:,1));
ymin = min(xy(:,2)); ymax = max(xy(:,2));
span = max(xy,[],1) - min(xy,[],1);
tol = max(max(span)*1e-10, 1e-12);
switch edge
    case 'left',   on = all(abs(xy_e([1 4],1) - xmin) <= tol);
    case 'right',  on = all(abs(xy_e([2 3],1) - xmax) <= tol);
    case 'bottom', on = all(abs(xy_e([1 2],2) - ymin) <= tol);
    case 'top',    on = all(abs(xy_e([3 4],2) - ymax) <= tol);
    otherwise,     on = false;
end
end

function fe = traction_edge_vector(xy_e, edge, tvec)
% Consistent nodal traction vector by 1D Gauss integration on chosen side.
gp = [-1/sqrt(3); 1/sqrt(3)];
fe = zeros(8,1);
for ig = 1:2
    s = gp(ig);
    switch edge
        case 'right',  xi = 1;  eta = s;
        case 'left',   xi = -1; eta = s;
        case 'top',    xi = s;  eta = 1;
        case 'bottom', xi = s;  eta = -1;
        otherwise, error('Unsupported edge.');
    end
    [N, dN_dxi, dN_deta] = shape_quad4(xi, eta);
    Nmat = Nmat_quad4(N);
    if abs(xi) == 1
        dX_dq = dN_deta * xy_e;
    else
        dX_dq = dN_dxi * xy_e;
    end
    fe = fe + (Nmat' * tvec) * norm(dX_dq); % [N/m]*[m] contribution
end
end

function [gps, w] = gauss2x2()
g = 1/sqrt(3);
gps = [-g -g; g -g; g g; -g g];
w = [1;1;1;1];
end

function [N, dN_dxi, dN_deta] = shape_quad4(xi, eta)
xi_a = [-1 1 1 -1];
eta_a = [-1 -1 1 1];
N = 0.25 * (1 + xi_a*xi) .* (1 + eta_a*eta);
dN_dxi = 0.25 * xi_a .* (1 + eta_a*eta);
dN_deta = 0.25 * eta_a .* (1 + xi_a*xi);
end

function [J, detJ, dN_dxdy] = jacobian_quad4(xy_e, dN_dxi, dN_deta)
J = [dN_dxi; dN_deta] * xy_e;
detJ = det(J);
if detJ <= 0
    error('Non-positive detJ.');
end
dN_dxdy = J \ [dN_dxi; dN_deta];
end

function B = Bmat_quad4(dN_dxdy)
B = zeros(3,8);
for a = 1:4
    ia = 2*a-1;
    B(1,ia) = dN_dxdy(1,a);
    B(2,ia+1) = dN_dxdy(2,a);
    B(3,ia) = dN_dxdy(2,a);
    B(3,ia+1) = dN_dxdy(1,a);
end
end

function Nmat = Nmat_quad4(N)
Nmat = zeros(2,8);
for a = 1:4
    ia = 2*a-1;
    Nmat(1,ia) = N(a);
    Nmat(2,ia+1) = N(a);
end
end

function [xy, conn] = gen_quad_mesh(L, W, nx, ny)
% Structured rectangular Quad4 mesh, CCW node ordering.
xs = linspace(-0.5*L, 0.5*L, nx+1);
ys = linspace(-0.5*W, 0.5*W, ny+1);
[X,Y] = meshgrid(xs, ys);
xy = [X(:), Y(:)];

conn = zeros(nx*ny,4);
eid = 0;
% MATLAB linear indexing is column-major for X(:),Y(:):
% node id at (ix,iy) is (ix-1)*(ny+1) + iy.
for i = 1:nx
    for j = 1:ny
        n1 = (i-1)*(ny+1) + j;      % bottom-left
        n2 = i*(ny+1) + j;          % bottom-right
        n3 = i*(ny+1) + (j+1);      % top-right
        n4 = (i-1)*(ny+1) + (j+1);  % top-left
        eid = eid + 1;
        conn(eid,:) = [n1 n2 n3 n4];
    end
end
end

function write_csv(fname, header, M)
fid = fopen(fname,'w');
fprintf(fid,'%s\n',header);
fclose(fid);
dlmwrite(fname, M, '-append', 'delimiter', ',', 'precision', '%.10e');
end
