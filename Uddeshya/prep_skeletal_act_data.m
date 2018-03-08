%given a skeletal file name it will return a struct with action type and
%the joint 3d cordinates of people in that file separately for all the
%frames present

fname = 'S001C001P001R001A001.skeleton';
joints_3d = get_only_action(fname);
