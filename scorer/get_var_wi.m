function err = get_var_wi(yFitObj, batch_id_funhan)
    tol = 1e-11;

    assert(isa(yFitObj,'yFit'),'get_mse() takes only yFit objects as input');
    
    id = batch_id_funhan(yFitObj);
    
    assert(length(id) == length(yFitObj.Y),'Unsatisfied Requirement: length(id) == length(yFitObj.Y)');
        
    [~,Y_wi] = splitVar(yFitObj.Y, id);
    [~,yfit_wi] = splitVar(yFitObj.yfit, id);
    
    assert(mean(Y_wi) < tol, 'Biased Y_wi returned. This shouldn''t happen.');
    assert(mean(yfit_wi) < tol, 'Biased yfit_wi returned. This shouldn''t happen.');
    % because Y_wi and yfit_wi are unbiased, we can just use var to compute
    % the corresponding MSE fraction
    err = var(Y_wi - yfit_wi,1);
end