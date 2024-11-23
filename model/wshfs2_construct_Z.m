function Z = wshfs2_construct_Z(X_cells, alpha, betas)

for j = 1:size(X_cells,2)
    for k = 1:size(X_cells,2)
        if j==k
            w = betas(j);
        else
            w = alpha;
        end
        
        z{j,k} = w*X_cells{1,j}'*X_cells{1,k};
    end
end

Z = cell2mat(z);
