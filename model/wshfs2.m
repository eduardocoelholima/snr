function [U,D] = wshfs2(X_cells,alpha,betas)
% WSHFS implementation
% input: X original input spaces
% output: X2 projected U-space

% constructs Z matrix such that X = u * Z * u'
Z = wshfs2_construct_Z(X_cells,alpha,betas);

% calculates the eigenvalues D and eigenvectors U
rand('twister',5489);
[U,D] = eig(Z);

% sort by eigenvalue magnitude, descending, prints explained variance
[d,ind] = sort(diag(D));
D = D(ind,ind);
U = U(:,ind);
