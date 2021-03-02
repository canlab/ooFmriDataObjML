% F1 score for binary targets
function err = get_f1(yFitObj)
    assert(isa(yFitObj,'yFit'),'get_f1() takes only yFit objects as input');

    true_positive = yFitObj.Y.*yFitObj.yfit;
    precision = sum(true_positive)/sum(yFitObj.yfit);
    
    recall = sum(true_positive)/sum(logical(yFitObj.Y)); % also known as the true positive rate
    
    f1 = 2 * precision * recall / (precision + recall);
    
    err = f1;
end