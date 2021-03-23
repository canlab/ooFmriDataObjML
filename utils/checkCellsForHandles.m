function hasHandles = checkCellsForHandles(obj)
    assert(isa(obj, 'cell'), sprintf('cell expected but found %s.', class(obj)));
    
    hasHandles = false;
    for i = 1:length(obj)
        if isa(obj{i},'cell')
            hasHandles = checkCellsForHandles(obj{i});
        elseif isa(obj{i}, 'handle')
            hasHandles = true;
            return;
        end
    end
end