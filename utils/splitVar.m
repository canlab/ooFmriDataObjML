% splits X into within and between fractions, such that X = bt + wi + bias, 
% bias = mean(X)*ones(length(X),1), bt is uniform for any index label in 
% id, and that uniform value is the offset of the indicated group from 
% mean(X), while wi is the element-by-element deviation from bias + bt of 
% each element in X.
function [bt, wi, bias] = splitVar(X,id)
    tol = 1e-11;

    if isvector(X)
        X = X(:);
    else 
        assert(ismatrix(X)); 
    end
    
    assert(length(id) == size(X,1),'dat must be length(id) x m');
    
    X = double(X);
    
    blk = dummyvar(categorical(id));
    meanmat = blk/(blk'*blk)*blk';
    
    wi = X - meanmat*X;
    bt = meanmat*X;
    bt = bt - mean(X,1);
    bias = mean(X,1)*ones(length(X),1);
    
    assert(all(wi + bt + bias - X < tol),'Variance splitting failed, X != bt + wi + bias.')
end