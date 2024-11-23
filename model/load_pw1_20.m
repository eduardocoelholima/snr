% headers = original input strings
% XS = preprocessed inputs
% XC = inputs with 0-mean columns
% XCN = inputs column-normalized
% XRN = inputs row-normalized

function [XS, gnd, XC, XCN, XRN, headers] = load_pw1_20()

% load ground truth labels
datapath = '../data/';
gnd = load(strcat(datapath,'pw1_386_api_target.dat'));

disp(['Loaded feature space gnd, n=',num2str(size(gnd,1)),' p=1/',num2str(size(gnd,2))]);

% filenames and identifiers
x_ids = {'sw2v','stag','smashup','stfidf','sprovider','slda','mtag','mlda','sine'};
x_files = {'pw1_386_api_w2v.dat', ...
    'pw1_386_api_tag.dat', ...
    'pw1_386_api_mashup.dat', ...
    'pw1_386_api_tfidf.dat', ...
    'pw1_386_api_provider.dat', ...
    'pw1_386_api_lda20.dat', ...
    'pw1_6366_mashup_tag.dat', ...
    'pw1_6366_mashup_lda.dat', ...
    'pw1_386_sine.dat', ...
    };

for k = 1:size(x_files,2)
    %read files
    XS(k).id = x_ids{k};
    XS(k).x = load(strcat(datapath,x_files{k}));
        
    %remove 0 columns, normalize columns on non-mashup related features
    previous_p = size(XS(k).x,2);
    
    % null column removal
    woot = find(sum(XS(k).x,1)~=0);
    if k ~= [3 7 8] %mashup-related features
        XS(k).x = XS(k).x(:,woot);
    end
        
    disp(['Loaded feature space ',XS(k).id,', n=',num2str(size(XS(k).x,1)),' p=',num2str(size(XS(k).x,2)),'/',num2str(previous_p)])
end

for s = 1:size(XS,2)
    XC(s).x = XS(s).x - mean(XS(s).x);
    XC(s).id = XS(s).id;
    
    % null column removal
    woot = find(sum(XC(s).x,1)~=0);
    if k == [3 7 8] %mashup-related features
        XC(k).x = XC(k).x(:, woot);
    end
end

% column normalization - not done on tfidf
for s = 1:size(XS,2)
    if s ~= [4]
        XCN(s).x = normalize(XS(s).x,1,'norm');
    else
        XCN(s).x = XS(s).x;
    end
    XCN(s).id = XS(s).id;
end

% normalize w2v
XS(1).x = normalize(XS(1).x,2,'norm');


% row normalized input - used for kmeans using cosine similarity
for s = 1:size(XS,2)
    if s == []
        XRN(s).x = normalize(XS(s).x,2,'norm');
    else
        XRN(s).x = XS(s).x;
    end
    XRN(s).id = XS(s).id;
end


% loads headers
header_ids = {'s','sdesc','stag','sgnd','tag','m','mtag','provider','smashup'};
header_files = {'pw1_386_api_names.dat', ...
    'pw1_386_api_descriptions.dat', ...
    'pw1_386_api_tag_names.dat', ...
    'pw1_386_api_target_names.dat', ...
    'pw1_477_tag_names.dat', ...
    'pw1_6366_mashup_names.dat', ...
    'pw1_6366_mashup_tag_names.dat', ...
    'pw1_4894_provider_names.dat', ...
    'pw1_386_api_mashup_names.dat', ...
    };
for k = 1:size(header_files,2)
    headers(k).id = header_ids{k};
    file = fopen(strcat(datapath,header_files{k}));
    cell = textscan(file,'%s','delimiter','\n');
    fclose(file);
    headers(k).h = string(cell{1});
end
