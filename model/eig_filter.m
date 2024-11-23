function [X_prime,explained] = eig_filter(X,D,U,k)
% function eig_filter
% projects X into U-subspace (k-dim limited)
% X2 = X * U, where U is the eigenvector matrix of Z
% refer to paper to see how Z is constructed
% 
% X input original space
% D eigenvalues
% U eigenvectors
% k cutoff (number of principal components to keep)

% matlab can may order eigenvectors ascending or descending
if (D(1:1)>D(end:end))
    U_prime = U(:,1:k);
    explained = sum(sum(D(:,1:k)))/sum(sum(D));
else
    U_prime = U(:,end-k:end);
    explained = sum(sum(D(:,end-k:end)))/sum(sum(D));
end

X_prime = X*U_prime;
