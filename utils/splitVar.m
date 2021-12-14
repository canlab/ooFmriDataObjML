% splits X into within and between fractions, such that X = bt + wi + bias, 
% bias = mean(X)*ones(length(X),1), bt is uniform for any index label in 
% id, and that uniform value is the offset of the indicated group from 
% mean(X), while wi is the element-by-element deviation from bias + bt of 
% each element in X.
function [bt, wi, bias] = splitVar(X,id)
    tol = 1e-8;

    if isvector(X)
        X = X(:);
    else 
        assert(ismatrix(X)); 
    end
    
    nans = isnan(X);
    if any(nans)
        X(nans) = nanmean(X);
        warning('%d nan''s found, replacing with mean', sum(nans));
    end
    
    assert(length(id) == size(X,1),'dat must be length(id) x m');
    
    X = double(X);
    
    blk = dummyvar(categorical(id));
    
    %fprintf('Multiplying large matrices...\n')
    bt = blk/(blk'*blk)*blk'*X; % bt + bias
    wi = X - bt;
    bt = bt - mean(X,1); % remove bias term
    bias = repmat(mean(X,1),size(X,1),1);
    
    SSident = wi + bt + bias - X;
    assert(all(SSident(:) < tol),'Variance splitting failed, X != bt + wi + bias.')
end
