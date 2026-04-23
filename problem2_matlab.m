% AE510 Project - Problem 2 (2D panel, Galerkin FEM Quad4)
% Condensed MATLAB version in lecture/HW8 style (SI units).
clear; clc;

%% Geometry and properties (SI)
IN_TO_M = 0.0254;
L = 56.0 * IN_TO_M;                  % m
W = 4.0  * IN_TO_M;                  % m
t = 0.5  * IN_TO_M;                  % m
E = 70e9;                            % Pa
nu = 0.3;

p_ref = 1e6;                         % Pa
tmag = p_ref * t;                    % N/m line traction

% ~10, ~100, ~1000 elements
mesh_levels = [5 2; 20 5; 50 20];
case_tags = {'case1','case2','case3'};
case_titles = {'Case 1','Case 2','Case 3'};

%% Output folder
this_file = mfilename('fullpath');
[this_dir,~,~] = fileparts(this_file);
out_dir = fullfile(this_dir, 'figures', 'panel', 'matlab');
if exist(out_dir, 'dir') ~= 7
    mkdir(out_dir);
end

% results table: [case_id, nelem, max_u, max_abs_sx, max_abs_sy, max_vm]
all_rows = [];
conv = cell(3,1);
for i = 1:3
    conv{i} = [];
end

fprintf('Problem 2 (Quad4) summary:\n');
for im = 1:size(mesh_levels,1)
    nx = mesh_levels(im,1);
    ny = mesh_levels(im,2);
    [xy, conn] = gen_quad_mesh(L, W, nx, ny);
    nelem = size(conn,1);

    for icase = 1:3
        % Case object in compact form: one edge traction + Dirichlet set.
        [loads, bc] = build_case(icase, xy, tmag);
        out = solve_quad_case(xy, conn, E, nu, t, loads, bc);

        max_u = max(out.u_mag);
        max_abs_sx = max(abs(out.sig_elem(:,1)));
        max_abs_sy = max(abs(out.sig_elem(:,2)));
        max_vm = max(out.vm_elem);

        fprintf('  %s, nelem=%d: max_u=%.4e m, max|sx|=%.4e Pa, max|sy|=%.4e Pa, max_vm=%.4e Pa\n', ...
            case_titles{icase}, nelem, max_u, max_abs_sx, max_abs_sy, max_vm);

        all_rows = [all_rows; icase, nelem, max_u, max_abs_sx, max_abs_sy, max_vm]; %#ok<AGROW>
        conv{icase} = [conv{icase}; nelem, max_abs_sx, max_abs_sy, max_vm]; %#ok<AGROW>

        % optional run-level node/element exports
        run_tag = sprintf('%s_n%d', case_tags{icase}, nelem);
        node_tab = [(1:size(xy,1))', xy, out.u_nodes, out.u_mag];
        elem_tab = [(1:nelem)', conn, out.sig_elem, out.vm_elem];
        write_csv(fullfile(out_dir, [run_tag '_node.csv']), ...
            'node,x_m,y_m,ux_m,uy_m,u_mag_m', node_tab);
        write_csv(fullfile(out_dir, [run_tag '_elem.csv']), ...
            'element,n1,n2,n3,n4,sigma_x_pa,sigma_y_pa,tau_xy_pa,sigma_vm_pa', elem_tab);
    end
end

%% Write combined outputs
write_csv(fullfile(out_dir, 'problem2_all_results.csv'), ...
    'case_id,nelem,max_u_m,max_abs_sigma_x_pa,max_abs_sigma_y_pa,max_sigma_vm_pa', all_rows);

fid = fopen(fullfile(out_dir,'problem2_convergence.txt'),'w');
for icase = 1:3
    C = sortrows(conv{icase},1);
    fprintf(fid, '%s\n', case_titles{icase});
    fprintf(fid, 'nelem,max_abs_sigma_x_pa,max_abs_sigma_y_pa,max_sigma_vm_pa\n');
    for i = 1:size(C,1)
        fprintf(fid, '%d,%.8e,%.8e,%.8e\n', C(i,1), C(i,2), C(i,3), C(i,4));
    end
    fprintf(fid,'\n');
end
fclose(fid);

%% Local functions
function [loads, bc] = build_case(icase, xy, tmag)
% Case definitions intentionally match the Python driver loading logic.
switch icase
    case 1
        loads = struct('edge','right','tvec',[tmag; 0.0]);
        bc = dirichlet_bcs_for_edge(xy, 'left', [1 2], 0.0);
    case 2
        loads = struct('edge','right','tvec',[tmag/sqrt(2); tmag/sqrt(2)]);
        bc = dirichlet_bcs_for_edge(xy, 'left', [1 2], 0.0);
    case 3
        loads = struct('edge','bottom','tvec',[0.0; -tmag]);
        bc1 = dirichlet_bcs_for_edge(xy, 'left', [1 2], 0.0);
        bc2 = dirichlet_bcs_for_edge(xy, 'right', [1 2], 0.0);
        bc = [bc1; bc2];
end
end

function out = solve_quad_case(xy, conn, E, nu, t, loads, bc)
Nn = size(xy,1); Ne = size(conn,1); dof = 2;
ndof = Nn*dof;

[gps, w] = gauss2x2();
D = D_plane_stress(E, nu);

% Triplet storage for sparse global stiffness assembly.
I = zeros(Ne*64,1); J = I; V = I;
nz = 0;
F = zeros(ndof,1);

for e = 1:Ne
    nodes = conn(e,:);
    xy_e = xy(nodes,:);
    Ke = zeros(8,8); fe = zeros(8,1);

    % Domain integration of Ke using 2x2 Gauss quadrature.
    for ig = 1:4
        xi = gps(ig,1); eta = gps(ig,2);
        [N, dN_dxi, dN_deta] = shape_quad4(xi, eta);
        [~, detJ, dN_dxdy] = jacobian_quad4(xy_e, dN_dxi, dN_deta);
        B = Bmat_quad4(dN_dxdy);
        Ke = Ke + (B' * D * B) * detJ * t * w(ig);
    end

    % Add consistent edge traction vector only for boundary elements.
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

% Final sparse global stiffness.
K = sparse(I(1:nz), J(1:nz), V(1:nz), ndof, ndof);

% BC solve (static condensation)
c_dofs = (bc(:,1)-1)*dof + bc(:,2);
c_vals = bc(:,3);
% Preserve first BC value when duplicate constraints exist.
[c_dofs, ia] = unique(c_dofs, 'stable');
c_vals = c_vals(ia);
all_dofs = (1:ndof).';
free_dofs = setdiff(all_dofs, c_dofs);

u = zeros(ndof,1);
u(c_dofs) = c_vals;
Ff = F(free_dofs) - K(free_dofs, c_dofs) * u(c_dofs);
u(free_dofs) = K(free_dofs, free_dofs) \ Ff; % reduced solve

% postprocessing at centroid
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
    % One stress sample per element at centroid (xi=0, eta=0).
    [~, dN_dxi, dN_deta] = shape_quad4(0.0, 0.0);
    [~, ~, dN_dxdy] = jacobian_quad4(xy_e, dN_dxi, dN_deta);
    B = Bmat_quad4(dN_dxdy);
    sig = D * (B * ue);
    sig_elem(e,:) = sig.';
    sx = sig(1); sy = sig(2); txy = sig(3);
    vm_elem(e) = sqrt(sx^2 - sx*sy + sy^2 + 3*txy^2);
end

u_nodes = reshape(u, 2, []).';
out.u_nodes = u_nodes;
out.u_mag = sqrt(sum(u_nodes.^2,2));
out.sig_elem = sig_elem;
out.vm_elem = vm_elem;
end

function bc = dirichlet_bcs_for_edge(xy, edge, dof_ids, val)
% Build [node,dof,value] rows for all nodes on selected boundary edge.
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
% Detect whether current element owns the requested global boundary edge.
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
% Consistent edge load vector: integrate N^T * t over selected element side.
gp = [-1/sqrt(3); 1/sqrt(3)];
fe = zeros(8,1);
for ig = 1:2
    s = gp(ig);
    switch edge
        case 'right',  xi = 1.0;  eta = s;
        case 'left',   xi = -1.0; eta = s;
        case 'top',    xi = s;    eta = 1.0;
        case 'bottom', xi = s;    eta = -1.0;
        otherwise, error('Unsupported edge.');
    end
    [N, dN_dxi, dN_deta] = shape_quad4(xi, eta);
    Nmat = Nmat_quad4(N);
    if abs(xi) == 1
        dX_dq = dN_deta * xy_e;
    else
        dX_dq = dN_dxi * xy_e;
    end
    Jline = norm(dX_dq);
    fe = fe + (Nmat' * tvec) * Jline; % [N/m]*[m] -> nodal force [N]
end
end

function [gps, w] = gauss2x2()
g = 1/sqrt(3);
gps = [-g -g; g -g; g g; -g g];
w = [1;1;1;1];
end

function D = D_plane_stress(E, nu)
D = (E/(1-nu^2)) * [1 nu 0; nu 1 0; 0 0 (1-nu)/2];
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
% Structured rectangular Quad4 mesh, node order is counter-clockwise.
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
