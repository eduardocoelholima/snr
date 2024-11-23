function AC = baseline_experiment(varargin)

XS = varargin{1};
gnd = varargin{2};
spaces = varargin{3};
distance = varargin{4};
k = varargin{5};
verbose = varargin{6};
headers = varargin{7};

for s = 1:size(spaces,2)
    X_cells{s} = XS(spaces(s)).x;
end

X = cell2mat(X_cells);
rng('default');
label = litekmeans(X,k,'Replicates',10,'Distance',distance);
result = bestMap(gnd,label);
AC = length(find(gnd == result))/length(gnd);

if verbose == 1
    for cluster = 0:k-1
        selected = find(result == cluster);
        fprintf('\ncluster %d: %d services\n',cluster,size(selected,1));
        for i = 1:size(selected,1)
            s = selected(i);
            fprintf('%d: %s, gnd=%s(%d), tags=%s, mashups=%s\n', ...
                s,headers(1).h(s),headers(3).h(s),gnd(s),headers(2).h(s),headers(8).h(s));
        end
    end
end