% function main()
tic

% loads data files
[XS gnd XC XCN XRN headers] = load_pw1_20();

% calculates metapaths similarity matrices %pathcount=0/pathsim=1
theta = [0.1667 0.1667 0.1667 0.1667 0.1667 0.1667];
[M MS] = prep_metapaths(XS,theta,1,zeros(size(gnd))); 

% % baseline experiments (cosine,kmeans), litekmeans
% model = "baseline ";
% combinations = {[2] [1] [4] [1 2] [4 2] [6]};
% k = 20;
% verbose = 0;
% for c = 1:size(combinations,2)
%     desc = '';
%     spaces = combinations{c};
%     for s = 1:size(spaces,2)
%         desc = strcat(desc,'+',XS(spaces(s)).id);
%     end
%     ac = baseline_experiment(XS,gnd,spaces,'cosine',k,verbose,headers);
%     fprintf('%s%s: AC=%.2f\n',model,desc,ac);
% end
% 
% % wshfs2 experiments
% model = "wshfs2 ";
% betas = [0.85057673 0.14785474 0.00156853 0.]; % tfidf, tags, mashup, provider
% ks = [20 40 60 80 100 120]; %principal components
% alphas = [0:0.2:1];%inter feature covariance weight
% combinations = {[4 2 3 5]};
% distance = 'cosine';
% sampleset = [1:76; 77:152; 153:228; 229:304; 305:380];
% for c = 1:size(combinations,2)
%     desc = '';
%     spaces = combinations{c};
%     for s = 1:size(spaces,2)
%         desc = strcat(desc,'+',XS(spaces(s)).id);
%     end
%     for alpha = alphas
%         for k = ks
%             ACS = [];
%             for s = 1:size(sampleset,1)
%                 samples = sampleset(s,:);
%                 ac1 = wshfs2_experiment(XC,gnd,spaces,k,alpha,betas,distance,samples);
%                 ACS = [ACS ac1];
%             end
%             ac = mean(ACS);
%             fprintf('%s%s: k=%d b=optimum a=%.2f AC=%.2f\n',model,desc,k,alpha,ac);
%         end
%     end
% end

% hsn2 experiments
combinations = {[4 2 5]}; % tfidf, tags, provider
alphas = [0.01];
betas = [0.5 0 0.5];

% sensitivity analysis
% alphas = [0.0001 0.001 0.01 0.1 1];

% sensitivity analysis
% betas = [0.0 1.0 0 ]
% betas = [0.1 0.9 0 ]
% betas = [0.3 0.7 0 ]
% betas = [0.5 0.5 0 ]
% betas = [0.7 0.3 0 ]
% betas = [0.9 0.1 0 ]
% betas = [1.0 0.0 0 ]
% --------------
% betas = [0.0 0 1.0 ]
% betas = [0.1 0 0.9 ]
% betas = [0.3 0 0.7 ]
% betas = [0.5 0 0.5 ]
% betas = [0.7 0 0.3 ]
% betas = [0.9 0 0.1 ]
% betas = [1.0 0 0.0 ]

% combinations = {[1 2 5]}; % % w2v, tags, mashups, provider
% betas = [7.79140627e-01 2.18529565e-01 2.32980839e-03 9.99363352e-17];
% alphas = [0.01];
% betas = [0.4 0 0.6];
model = "hsn2 ";
distance = 'cosine'; 
sampleset = [1:76; 77:152; 153:228; 229:304; 305:380];
ks = [20];
w = [0 9.99994439e-01 2.40636815e-03 1.27740238e-03 5.74226786e-04 0]; %meta-path weighting
M2 = w(1)*MS(1).sim + w(2)*MS(2).sim + w(3)*MS(3).sim + w(4)*MS(4).sim + ...
    w(5)*MS(5).sim + w(6)*MS(6).sim;
for c = 1:size(combinations,2)
    desc = '';
    spaces = combinations{c};
    for s = 1:size(spaces,2)
        desc = strcat(desc,'+',XS(spaces(s)).id);
    end
    for alpha = alphas
        for k = ks
            ACS = [];
            for s = 1:size(sampleset,1)
                samples = sampleset(s,:);
                [metric1 X_prime] = hsn2_experiment(XC,gnd,spaces,k,alpha,betas,M2,distance,samples);
                ACS = [ACS metric1.Accuracy];
                PRS = [ACS metric1.Precision];
                RCS = [ACS metric1.Recall];
            end
            ac = mean(ACS);
            pr = mean(PRS);
            rc = mean(RCS);
            fprintf('%s%s: k=%d b=optimum a=%.2f AC=%.2f PR=%.2f RC=%.2f\n',model,desc,k,alpha,ac,pr,rc);
        end
    end
    samples = 1:size(XC(4).x,1);
    [ac1 X_prime] = hsn2_experiment(XC,gnd,spaces,k,alpha,betas,M2,distance,samples);
end

%normalize embeddings
x = normalize(X_prime,2,'norm');


toc