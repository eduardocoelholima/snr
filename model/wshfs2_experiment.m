function AC = wshfs2_experiment(varargin)

XS = varargin{1};
gnd = varargin{2};
spaces = varargin{3};
k = varargin{4};
alpha = varargin{5};
betas = varargin{6};
distance = varargin{7};
samples = varargin{8};

for s = 1:size(spaces,2)
    X_cells{s} = XS(spaces(s)).x(samples,:);
end

X = cell2mat(X_cells);
% size(X)
[U,D] = wshfs2(X_cells,alpha,betas);
[X_prime,explained] = eig_filter(X,D,U,k);
rng('default');
if distance == 'cosine'
    X_prime = normalize(X_prime,1,'norm');
end
label = litekmeans(X_prime,18,'Replicates',10,'Distance',distance);
gnd = gnd(samples);
% size(gnd)
result = bestMap(gnd,label);
AC = length(find(gnd == result))/length(gnd);
