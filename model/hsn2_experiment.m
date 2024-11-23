function [AC X_prime] = hsn2_experiment(varargin)

XS = varargin{1};
gnd = varargin{2};
spaces = varargin{3};
k = varargin{4};
alpha = varargin{5};
betas = varargin{6};
m = varargin{7};
distance = varargin{8};
samples = varargin{9};

for s = 1:size(spaces,2)
    X_cells{s} = XS(spaces(s)).x(samples,:);
end
% 
% for s = 1:size(spaces,2)
%     X_cells{s} = XS(spaces(s)).x;
% end

X = cell2mat(X_cells);
% [U,D] = hsn2(X_cells,alpha,betas,m);
[U,D] = hsn2(X_cells,alpha,betas,m(samples,samples));
[X_prime,explained] = eig_filter(X,D,U,k);
rng('default');
label = litekmeans(X_prime,18,'Replicates',10,'Distance',distance);

gnd = gnd(samples);

result = bestMap(gnd,label);

% AC = length(find(gnd == result))/length(gnd); % use for accuracy only

conf = confusionmat(gnd,result);
metrics = multiclass_metrics_common(conf);
% AC = metrics.Accuracy; % Accuracy,Precision,Recall, F1score available
AC = metrics;