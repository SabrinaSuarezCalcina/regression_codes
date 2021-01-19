function [] = print_matrix(A, nome, folder)
fi = strcat(nome(1:end));
fid = fopen([folder '/' fi ], 'w+');

for i = 1:size(A, 1)
    fprintf(fid, '%.3f ', A(i,:));
    fprintf(fid, '\n');
end
fclose(fid);
end