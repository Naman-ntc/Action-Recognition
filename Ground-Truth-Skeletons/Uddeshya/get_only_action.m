function [joints3d] = get_only_action(fname)
%GET_ONLY_ACTION Summary of this function goes here
%   Detailed explanation goes here
    last_name = strsplit(fname, '/');
    idx = size(last_name,2);
    last_name = last_name{idx};
    joints3d.label = str2double(last_name(18:20));  
    
    fileid = fopen(fname);
    framecount = fscanf(fileid,'%d',1); % no of the recorded frames

    joints3d.frames=[]; % to store multiple skeletons per frame

    for f=1:framecount
        bodycount = fscanf(fileid,'%d',1); % no of observerd skeletons in current frame
        for b=1:bodycount
            clear body;
            body.bodyID = fscanf(fileid,'%ld',1); % tracking id of the skeleton
            %arrayint = fscanf(fileid,'%d',6); % read 6 integersls
            
            fscanf(fileid,'%d',6);
            %body.clipedEdges = arrayint(1);
            %body.handLeftConfidence = arrayint(2);
            %body.handLeftState = arrayint(3);
            %body.handRightConfidence = arrayint(4);
            %body.handRightState = arrayint(5);
            %body.isResticted = arrayint(6);
            %lean = fscanf(fileid,'%f',2);
            fscanf(fileid,'%f',2);
            %body.leanX = lean(1);
            %body.leanY = lean(2);
            %body.trackingState = fscanf(fileid,'%d',1);
             fscanf(fileid,'%d',1);

            body.jointCount = fscanf(fileid,'%d',1); % no of joints (25)
            joints=[];
            for j=1:body.jointCount
                jointinfo = fscanf(fileid,'%f',11);
                joint=[];

                % 3D location of the joint j
                joint.x = jointinfo(1);
                joint.y = jointinfo(2);
                joint.z = jointinfo(3);

                % 2D location of the joint j in corresponding depth/IR frame
                %joint.depthX = jointinfo(4);
                %joint.depthY = jointinfo(5);

                % 2D location of the joint j in corresponding RGB frame
                %joint.colorX = jointinfo(6);
                %joint.colorY = jointinfo(7);

                % The orientation of the joint j
                %joint.orientationW = jointinfo(8);
                %joint.orientationX = jointinfo(9);
                %joint.orientationY = jointinfo(10);
                %joint.orientationZ = jointinfo(11);

                % The tracking state of the joint j
                %joint.trackingState = fscanf(fileid,'%d',1);
                fscanf(fileid,'%d',1);

                body.joints(j)=joint;
            end
            joints3d.frames(f).bodies(b)=body;
        end
    end
    fclose(fileid);
end

