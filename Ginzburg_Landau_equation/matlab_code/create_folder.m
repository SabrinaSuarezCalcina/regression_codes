function create_folder(name)
if (~(exist(name,'dir')))
    mkdir(name);
end
end
