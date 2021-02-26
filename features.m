% obj = features(dat, metadata)
% this function extends double's to attach some metadata (e.g. subject
% block identifiers) to vectors. size(metadata,1) must equal size(data,1)
% size(data,1). In other words, you need an identifier for each row of your
% data.
%
% The object will persist across subscripting, subreferencing, and 
% concatenation but other operations will promote the object back to it's
% parent type double, erasing the metadata property. This should be fine
% for managing block identifiers for cross validation though.
%
% see here for details on the coding methods used below
% https://www.mathworks.com/help/matlab/matlab_oop/built-in-subclasses-with-properties.html

classdef features < double
    properties
        metadata;
    end
    
    methods
        function obj = features(dat, metadata)
            if nargin == 0
                dat = 0;
                metadata = [];
            elseif nargin == 1
                metadata = [];
            else
                [n1,m1] = size(dat);
                [n2,m2] = size(metadata);
                assert(n1 == n2, 'dat and metadata dimension mismatch');
            end
            
            obj = obj@double(dat);
            obj.metadata = metadata;
        end
        
        function sref = subsref(sref, s)
            switch s(1).type
                case '.'
                   switch s(1).subs
                      case 'metadata'
                         sref = sref.metadata;
                      case 'data'
                         d = double(sref);
                         if length(s)<2
                            sref = d;
                         elseif length(s)>1 && strcmp(s(2).type,'()')
                            sref = subsref(d,s(2:end));
                         end
                      otherwise
                         error('Not a supported indexing expression')
                   end
                case '()'
                    d = double(sref);
                    newd = subsref(d, s);

                    if size(sref.metadata,1) == size(sref,1)
                        newmd = subsref(sref.metadata, s);
                    else
                        newmd = sref.metadata;
                    end

                    sref = features(newd, newmd);
                case '{}'
                    error('Not a supported indexing expression');
            end
        end
        
        function obj = subsasgn(obj, s, b)
            switch s(1).type
                case '.'
                   switch s(1).subs
                      case 'metadata'
                         obj.metadata = b;
                      case 'data'
                         if length(s)<2
                            obj = features(b,obj.DataString);
                         elseif length(s)>1 && strcmp(s(2).type,'()')
                            d = double(obj);
                            newd = subsasgn(d,s(2:end),b);
                            obj = features(newd,obj.DataString);
                         end
                      otherwise
                         error('Not a supported indexing expression')
                   end
                case '()'
                    d = double(obj);
                    newd = subsasgn(d, s, b);

                    if length(obj.metadata) == length(obj)
                        newmd = subsasgn(obj.metadata, s, b);
                    else
                        newmd = obj.metadata;
                    end

                    obj = features(newd, newmd);
                case '{}'
                    error('Not a supported indexing expression')
            end
        end
        
        function obj = horzcat(varargin)
            d1 = cellfun(@double,varargin,'UniformOutput',false);
            data = horzcat(d1{:});
            newmd = eval(['cellfun(@(x1)(' class(varargin{1}.metadata) '(x1.metadata)),varargin,''UniformOutput'',false);']);
            newmd = horzcat(newmd{:});
            obj = features(data, newmd);
        end
      
        function obj = vertcat(varargin)
            d1 = cellfun(@double,varargin,'UniformOutput',false);
            data = vertcat(d1{:});
            newmd = eval(['cellfun(@(x1)(' class(varargin{1}.metadata) '(x1.metadata)), varargin,''UniformOutput'',false);']);
            newmd = vertcat(newmd{:});
            obj = features(data, newmd);
        end
    end
end