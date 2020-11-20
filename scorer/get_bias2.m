function err = get_bias2(yFitObj)
    assert(isa(yFitObj,'yFit'),'get_mse() takes only yFit objects as input');

    err = mean(yFitObj.yfit)*ones(length(yFitObj.yfit),1) - mean(yFitObj.Y)*ones(length(yFitObj.Y),1);
    err = mean(err.^2);
end
        