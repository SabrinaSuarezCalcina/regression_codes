classdef calc_regress_no_PH
    properties (SetAccess = public)              
    end
    methods (Static = true)
        function executar()                 
            beta = [1.750, 1.800, 1.850, 1.900, 1.950, 2.000, 2.050, 2.100, 2.150, 2.200, 2.250];
            
            for i = 1: length(beta)
            calc_regress_no_PH.form_vector(num2str(beta(i),'%.3f'))
            end
            number_samples = 3010;
            calc_regress_no_PH.calc_File_mat(number_samples, beta)
            calc_regress_no_PH.file_CSV()
            calc_regress_no_PH.file_CSV_Beta()          
        end
        
        function form_vector(beta)
            entry_path = ['../Matrix_Predator_Prey_Equation/Matrix/Param_beta' num2str(beta) '/U_grid'];
            output_path = ['../Vector/Param_beta' num2str(beta)];
            create_folder(output_path)          
            addpath(entry_path)
            fil = dir([entry_path '/*.dat']);
            for k = 1:length(fil)
                U_grid = load([entry_path '/' fil(k).name]);  % Matrix reading
                U_grid = U_grid';
                U_filt = U_grid(:);
                print_matrix(U_filt, fil(k).name, output_path) 
            end           
        end
        
        function calc_File_mat(number_samples, beta)
            output_path = '../Folder_mat/Predator_Prey_Equation';
            create_folder(output_path);                                                
            ii = 1; 
            celula = cell(number_samples,4);
            for n_beta = 1:length(beta)                                
                entry_path = ['../Vector/Param_beta' num2str(beta(n_beta),'%.3f')];
                folder_name = ['Beta_' num2str(beta(n_beta),'%.3f')];
                addpath(entry_path)
                file_na = dir([entry_path '/*.dat']);                                                       
                for i = 1:length(file_na)  
                    A = load([entry_path '/' file_na(i).name]);  % Vector reading
                    AA = A';
                    celula(ii,1) = {folder_name};
                    celula(ii,2) = {file_na(i).name(1:4)};
                    celula(ii,3) = {AA};
                    celula(ii,4) = {num2str(beta(n_beta),'%.3f')};                    
                    ii =  ii + 1;
                end                
            end
            save([output_path '/cell_Beta.mat'], 'celula', '-v7.3')  % Save cell
        end     
        function file_CSV() % TDA features
            entry_path = '../Folder_mat/Predator_Prey_Equation';
            output_path = '../Folder_CSV/Predator_Prey_Equation';
            create_folder(output_path)
            data = load([entry_path '/cell_Beta.mat']); % Load cell 
            Folder_name = data.celula(:,1);
            File_name = data.celula(:,2);  
            B = cell2mat(data.celula(:,3));
            T = table(Folder_name,File_name,B);
            writetable(T, [output_path '/TDA_Beta_features.txt'])   
        end
        function file_CSV_Beta() % Parameters values
            entry_path = '../Folder_mat/Predator_Prey_Equation';
            output_path = '../Folder_CSV/Predator_Prey_Equation';
            create_folder(output_path)
            data = load([entry_path '/cell_Beta.mat']); % Load cell   
            Folder_name = data.celula(:,1); 
            File_name = data.celula(:,2);  
            Beta = data.celula(:,4);
            T = table(Folder_name,File_name,Beta); 
            writetable(T, [output_path '/Beta_Parameter_value.txt'])                      
        end
    end
end
