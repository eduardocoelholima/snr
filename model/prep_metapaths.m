function [M Mset] = prep_metapaths(XS, theta, pathsim, M0)

SM = XS(3).x; % service-mashup
MS = SM';
MTm = XS(7).x; % mashup-mtag
TmM = MTm';
STs = XS(2).x; % service-tag
TsS = STs';
SPs = XS(5).x; % service-provider
PsS = SPs';

SLs = XS(6).x; % service-topics (lda of description)
LsS = SLs';
MLm = XS(8).x; % mashup-topics (lda of description)
LmM = MLm';

Mset(1).sim = SM*MS*SM*MS;
Mset(2).sim = SM*MTm*TmM*MS;
Mset(3).sim = SM*MS*STs*TsS*SM*MS;
Mset(4).sim = SM*MS*SPs*PsS*SM*MS;
Mset(5).sim = SPs*PsS;
Mset(6).sim = STs*TsS;
% M(7).sim = SM*MLm*LmM*MS;
% M(8).sim = SM*MS*SLs*LsS*SM*MS;
Mset(1).length = 4;
Mset(2).length = 4;
Mset(3).length = 6;
Mset(4).length = 6;
Mset(5).length = 2;
Mset(6).length = 2;

%just stores counts for apis and metapaths in a more friendly variable
api_count = size(XS(1).x,1);
metapath_count = size(Mset,2);

%calculates strengths and importances for each metapath
%in our case, graph is undirected, so in a link R(A,B), O(A)=I(B)
alpha = 0.2; %alpha = [0,1]
Mset(1).strength = 1 / ( sum(SM(:)) + sum(MS(:)) + sum(SM(:)) + sum(MS(:)))^alpha;
Mset(2).strength = 1 / ( sum(SM(:)) + sum(MTm(:)) + sum(TmM(:)) + sum(MS(:))^alpha );
Mset(3).strength = 1 / ( sum(SM(:)) + sum(MS(:)) + sum(STs(:)) + sum(TsS(:)) + sum(SM(:)) + sum(MS(:))^alpha );
Mset(4).strength = 1 / ( sum(SM(:)) + sum(MS(:)) + sum(SPs(:)) + sum(PsS(:)) + sum(SM(:)) + sum(MS(:))^alpha );
Mset(5).strength = 1 / ( sum(SPs(:)) + sum(PsS(:)) )^alpha;
Mset(6).strength = 1 / ( sum(STs(:)) + sum(TsS(:)) )^alpha;

for i = 1:metapath_count
    Mset(i).importance = exp(-Mset(i).length.*Mset(i).strength);
end


%convert from pathcount to pathsim
if pathsim == 1
    for x = 1:metapath_count
        for i = 1:api_count
            d1(i,:) = Mset(x).sim(i,i);
        end
        for j = 1:api_count
            d2(:,j) = Mset(x).sim(j,j);
        end
        d = d1 + d2;
        zeros = find(d == 0);
        d(zeros) = 1;
        
        Mset(x).sim = 2 * Mset(x).sim ./d;
    end
end

%averages MS into a single similarity matrix M
% M = zeros(1100);
M = M0;
for x = 1:metapath_count
    M = M + ( theta(x) .* Mset(x).sim );
end
