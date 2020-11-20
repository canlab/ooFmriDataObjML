function err = get_var(yFitObj)
    assert(isa(yFitObj,'yFit'),'get_mse() takes only yFit objects as input');

    err = var(yFitObj.yfit - yFitObj.Y,1);
end
        