clear

% joint positions recorded for training the newtwork 
fid = fopen('joint.txt','r');
T = fscanf(fid, '%f');
fclose(fid);
for l= 2:numel(T)
    T1(ceil((l-1)/63), mod((l-2),63) + 1) = T(l);
end

for i = 1: floor(size(T,1)/63)

    for j = 1:63
        if mod((j-1),3) == 0
            x(i,ceil(j/3)) = T1(i,j);
        elseif mod((j-1),3) == 1
            y(i,ceil(j/3)) = T1(i,j);
        elseif mod(j,3) ==0
            z(i,ceil(j/3)) = T1(i,j);
        end
    end
    
    x_max(i) = max(x(i,:));
    y_max(i) = max(y(i,:));
    z_max(i) = max(z(i,:));
    x_min(i) = min(x(i,:));
    y_min(i) = min(y(i,:));
    z_min(i) = min(z(i,:));
    
    M=32;
    % Bounding box computation
    l_voxel(i) = (max([x_max(i)-x_min(i), y_max(i)-y_min(i), z_max(i)-z_min(i)])/M);
    
    center(i,1) = x_min(i)+(x_max(i)-x_min(i))/2;
    center(i,2) = y_min(i)+(y_max(i)-y_min(i))/2;
    center(i,3) = z_min(i)+(z_max(i)-z_min(i))/2;
%     coordinate transformation
    x(i,1:end) = x(i,1:end)-center(i,1);
    y(i,1:end) = y(i,1:end)-center(i,2);
    z(i,1:end) = z(i,1:end)-center(i,3);
    
    % transformed Y_n as in paper
    transformed_x(i,1:21) = x(i,1:21) / (M*l_voxel(i)) + 0.5;
    transformed_y(i,1:21) = y(i,1:21) / (M*l_voxel(i)) + 0.5;
    transformed_z(i,1:21) = z(i,1:21) / (M*l_voxel(i)) + 0.5;
    
    for j = 1:63
        if mod((j-1),3) == 0
            tf_c(i,j) = transformed_x(i,ceil(j/3));
        elseif mod((j-1),3) == 1
            tf_c(i,j) = transformed_y(i,ceil(j/3));
        elseif mod(j,3) ==0
            tf_c(i,j) = transformed_z(i,ceil(j/3));
        end
    end

end
% store the transformed joint as the ground truth for the network
csvwrite('transformed_joint.txt',tf_c);
