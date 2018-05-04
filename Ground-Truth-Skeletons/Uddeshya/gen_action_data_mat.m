root_path = '../datasets/NTU/nturgb+d_skeletons/';

for i=1:60
    dirname='A0';
    if i<=9
        dirname=strcat(dirname,'0',num2str(i));
    else
        dirname=strcat(dirname, num2str(i));
    end
    mkdir(strcat('./data/', dirname))
    d_path = strcat(root_path, dirname, '/*skeleton');
    files = dir(d_path);
    for file = files'
        clear joints3d
        fname = strcat(root_path, dirname,'/',file.name);
        l_name = strsplit(file.name, '.');
        l_name = l_name{1};
        joints3d = get_only_action(fname);
        %display(l_name)
        save(strcat('./data/', dirname, '/', l_name, '.mat'), 'joints3d');
    end 
end