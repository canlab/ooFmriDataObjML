function one_minus_r = get_r(yFitObj)
    assert(isa(yFitObj,'yFit'),'get_r() takes only yFit objects as input');

    one_minus_r = 1 - corr(yFitObj.yfit(:), yFitObj.Y(:));
end