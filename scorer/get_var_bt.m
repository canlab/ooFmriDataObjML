function err = get_var_bt(yFitObj, batch_id_funhan)
    tol = 1e-11;

    assert(isa(yFitObj,'yFit'),'get_mse() takes only yFit objects as input');
    
    id = batch_id_funhan(yFitObj);
    
    assert(length(id) == length(yFitObj.Y),'Unsatisfied Requirement: length(id) == length(yFitObj.Y)');
        
    [Y_bt, ~] = splitVar(yFitObj.Y, id);
    [yfit_bt, ~] = splitVar(yFitObj.yfit, id);
    
    assert(mean(Y_bt) < tol, 'Biased Y_bt returned. This shouldn''t happen.');
    assert(mean(yfit_bt) < tol, 'Biased yfit_bt returned. This shouldn''t happen.');
    % because Y_wi and yfit_wi are unbiased, we can just use var to compute
    % the corresponding MSE fraction
    err = var(Y_bt - yfit_bt,1);
end