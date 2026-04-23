% AE510 Project - Problem 1 (3D truss, direct stiffness)
% Condensed MATLAB version in lecture/sample style (SI units).
clear; clc;

%% Preprocessing (geometry, properties, connectivity)
IN_TO_M = 0.0254;
LBF_TO_N = 4.4482216152605;

E = 70e9;                              % Pa
w_drone = 90.0 * LBF_TO_N;            % N
w_package = 5.0 * LBF_TO_N;           % N

w_outer = 4.0 * IN_TO_M;              % m
w_inner = 2.0 * IN_TO_M;              % m
t = 0.5 * IN_TO_M;                    % m

co = [  0,    0,    0;
       39,    0,    0;
       28, 32.5,    0;
      -28, 32.5,    0;
      -39,   0,     0;
      -15,-32.5,    0;
       15,-32.5,    0] * IN_TO_M;

% hub drop for true 3D truss response
co(1,3) = -2.0 * IN_TO_M;

e = [1 2;
     1 3;
     1 4;
     1 5;
     1 6;
     1 7;
     2 3;
     3 4;
     4 5;
     5 6;
     6 7;
     7 2];

Nel = size(e,1);
Nnodes = size(co,1);
dof = 3;

% per-element areas
A = zeros(Nel,1);
A(1:6) = w_inner * t;                 % spokes
A(7:12) = w_outer * t;                % rim

% structural weight check
Lel = zeros(Nel,1);
for i = 1:Nel
    Lel(i) = norm(co(e(i,2),:) - co(e(i,1),:));
end
rho_al = 2700.0;                      % kg/m^3
g = 9.80665;                          % m/s^2
V_total = sum(Lel .* A);
W_struct = rho_al * V_total * g;
fprintf('Problem 1 geometry check:\n');
fprintf('  Structural weight = %.3f N, limit = %.3f N, valid = %d\n\n', W_struct, w_drone, W_struct <= w_drone);

%% Output folder
this_file = mfilename('fullpath');
[this_dir,~,~] = fileparts(this_file);
out_dir = fullfile(this_dir, 'figures', 'truss', 'matlab');
if exist(out_dir, 'dir') ~= 7
    mkdir(out_dir);
end

%% Cases (same modeling intent as Python)
case_names = {'case1','case2','case3'};
case_desc = { ...
    'Case 1: hub package load', ...
    'Case 2: pull at node 1, hold at node 4', ...
    'Case 3: node 5 snagged, distributed thrust'};

for icase = 1:3
    F = zeros(Nnodes*dof,1);
    bc = []; % rows: [node, dof_index(1:x,2:y,3:z), prescribed_value]

    switch icase
        case 1
            % -package load at hub in z
            F((1-1)*dof + 3) = -w_package;
            % fix nodes 2..7 in ux,uy,uz
            for n = 2:7
                bc = [bc; n 1 0; n 2 0; n 3 0]; %#ok<AGROW>
            end

        case 2
            % +x pull at node 2 (1000 N)
            F((2-1)*dof + 1) = 1000.0;
            % node 5 fixed
            bc = [bc; 5 1 0; 5 2 0; 5 3 0];
            % node 2: uy,uz fixed
            bc = [bc; 2 2 0; 2 3 0];
            % node 1: uy fixed
            bc = [bc; 1 2 0];
            % nodes 3,4,6,7: uz fixed
            bc = [bc; 3 3 0; 4 3 0; 6 3 0; 7 3 0];

        case 3
            % +z distributed thrust on nodes 1,2,3,4,5,7
            load_nodes = [1 2 3 4 5 7];
            pnode = w_drone / 6.0;
            for n = load_nodes
                F((n-1)*dof + 3) = F((n-1)*dof + 3) + pnode;
            end
            % node 6 fixed
            bc = [bc; 6 1 0; 6 2 0; 6 3 0];
            % nodes 1,2,3,4,5,7: ux,uy fixed
            for n = [1 2 3 4 5 7]
                bc = [bc; n 1 0; n 2 0]; %#ok<AGROW>
            end
    end

    % Solve using one global K/F system + static condensation BC treatment.
    [u, strain, stress, reactions] = solve_truss3d_direct(co, e, A, E, F, bc);
    u_nodes = reshape(u, dof, []).';
    u_mag = sqrt(sum(u_nodes.^2,2));

    fprintf('%s\n', case_desc{icase});
    fprintf('  max nodal displacement = %.6e m\n', max(u_mag));
    fprintf('  stress range = [%.6e, %.6e] Pa\n', min(stress), max(stress));
    fprintf('  max |stress| = %.6e Pa\n\n', max(abs(stress)));

    % write node displacement csv
    node_table = [(1:Nnodes)', u_nodes, u_mag];
    node_csv = fullfile(out_dir, sprintf('%s_node_displacements.csv', case_names{icase}));
    write_csv(node_csv, 'node,ux_m,uy_m,uz_m,u_mag_m', node_table);

    % write element results csv
    elem_table = [(1:Nel)', e, strain, stress];
    elem_csv = fullfile(out_dir, sprintf('%s_element_results.csv', case_names{icase}));
    write_csv(elem_csv, 'element,node_i,node_j,strain,stress_pa', elem_table);

    % write summary txt
    txt_file = fullfile(out_dir, sprintf('%s_summary.txt', case_names{icase}));
    fid = fopen(txt_file,'w');
    fprintf(fid, 'max_nodal_deflection_m=%.8e\n', max(u_mag));
    fprintf(fid, 'min_element_stress_pa=%.8e\n', min(stress));
    fprintf(fid, 'max_element_stress_pa=%.8e\n', max(stress));
    fprintf(fid, 'max_abs_element_stress_pa=%.8e\n', max(abs(stress)));
    if ~isempty(bc)
        bc_dofs = sub2ind([Nnodes,dof], bc(:,1), bc(:,2));
        fprintf(fid, 'reaction_norm_paired_dofs=%.8e\n', norm(reactions(bc_dofs)));
    end
    fclose(fid);
end

%% Local functions
function [u, strain, stress, R] = solve_truss3d_direct(co, e, A, E, F, bc)
Nnodes = size(co,1);
Nel = size(e,1);
dof = 3;
K = zeros(Nnodes*dof, Nnodes*dof);

% Assemble global stiffness
for Ael = 1:Nel
    n = co(e(Ael,2),:) - co(e(Ael,1),:);
    L = norm(n);
    ns = n / L;
    k11 = (E * A(Ael) / L) * (ns' * ns);
    klocal = [k11, -k11; -k11, k11];

    % Element dof map [n1x n1y n1z n2x n2y n2z] into global indexing.
    edofs = [(e(Ael,1)-1)*dof + (1:dof), (e(Ael,2)-1)*dof + (1:dof)];
    K(edofs, edofs) = K(edofs, edofs) + klocal;
end

% Dirichlet BCs (static condensation)
all_dofs = (1:Nnodes*dof).';
if isempty(bc)
    c_dofs = [];
    c_vals = [];
else
    c_dofs = (bc(:,1)-1)*dof + bc(:,2);
    c_vals = bc(:,3);
    % Remove duplicate constraints while preserving first definition.
    [c_dofs, ia] = unique(c_dofs, 'stable');
    c_vals = c_vals(ia);
end
free_dofs = setdiff(all_dofs, c_dofs);

u = zeros(Nnodes*dof,1);
if ~isempty(c_dofs)
    u(c_dofs) = c_vals;
end
% Reduced system: Kff * uf = Ff - Kfc*uc.
Kff = K(free_dofs, free_dofs);
Ff = F(free_dofs) - K(free_dofs, c_dofs) * u(c_dofs);
u(free_dofs) = Kff \ Ff;

% Reactions
R = K*u - F;

% Postprocess element stress/strain
strain = zeros(Nel,1);
stress = zeros(Nel,1);
for i = 1:Nel
    n = co(e(i,2),:) - co(e(i,1),:);
    L = norm(n);
    ns = n / L;
    d1 = u((e(i,1)-1)*dof + (1:dof));
    d2 = u((e(i,2)-1)*dof + (1:dof));
    % 1D constitutive recovery along each member local axis.
    axial_ext = ns * (d2 - d1);
    strain(i) = axial_ext / L;
    stress(i) = E * strain(i);
end
end

function write_csv(fname, header, M)
% Lightweight csv writer used consistently across project scripts.
fid = fopen(fname,'w');
fprintf(fid,'%s\n',header);
fclose(fid);
dlmwrite(fname, M, '-append', 'delimiter', ',', 'precision', '%.10e');
end
