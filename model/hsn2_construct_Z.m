function Z = hsn2_construct_Z(X_cells, alpha, betas, m)


% calculates m-transformed covariances and weights them by w
for j = 1:size(X_cells,2)
    for k = 1:size(X_cells,2)
        if j==k
            w = betas(j);
        else
            w = alpha;
        end
        
        z{j,k} = w*(m*X_cells{1,j})'*(m*X_cells{1,k});
    end
end

Z = cell2mat(z);
