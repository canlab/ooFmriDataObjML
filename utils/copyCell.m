function newObj = copyCell(obj)
    assert(isa(obj, 'cell'), sprintf('cell expected but found %s.', class(obj)));
    
    newObj = cell(size(obj));
    for i = 1:length(obj)
        if isa(obj{i},'cell')
            newObj{i} = copyCell(obj{i});
        elseif isa(obj{i}, 'matlab.mixin.Copyable')
            newObj{i} = copy(obj{i});
        elseif isa(obj{i}, 'handle')
            warning('%s object found inside cell structure during deep copy invocation. This will not be copied.',class(obj{i}));
        end
    end
end