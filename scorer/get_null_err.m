function err = get_null_err(yFitObj)
    assert(isa(yFitObj,'yFit'),'get_nmse() takes only yFit objects as input');

    err = mean((yFitObj.yfit_null - yFitObj.Y).^2);
end
