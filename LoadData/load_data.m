data_dir = 'C:\Users\User\Documents\Mafaat_new_Topics\Shirim\DATA\';
dir_lst = ls([data_dir 'POS*']);
for ii=1:size(dir_lst,1)
    dir_path = [rmv_spaces([data_dir dir_lst(ii,:)])  '\Patch_1'];
    if isempty(ls([dir_path '\*.dat']))
        continue
    end
    Name = dir_lst(ii,:);
    if ~(contains(Name,'SPECH') || contains(Name,'SPEECH'))
        continue
    end
    if ~exist(['Data/' Name],'dir')
        mkdir(['Data/' Name])
    end
    com_pre = 'CorrelationResult_';
    com_post = '_';
    fs=8e3;
    objxyc = Class_XYC_Data_v4([],[],[]);
    objxyc = objxyc.LoadMultyfiles_by_Folder_ComPre_ComPost_CounterSt_nRois(...
        dir_path,com_pre,com_post,0,1);


    objxyc.dX(isnan(objxyc.dX) | objxyc.C<.7)=0;
    objxyc.dY(isnan(objxyc.dY) | objxyc.C<.7)=0;
    
    [U,S,V]=svds([objxyc.dX objxyc.dY],2);
    sig = [objxyc.dX objxyc.dY U]; 
    sig = sig./max(abs(sig));
    % sound(U(1e6+(1:fs*10),2),fs)
    
    audiowrite(['Data/' Name '/sig.wav'], sig, fs);
%     audiowrite(['Data/' Name '/sig_x.wav'], sig(:,1), fs);
%     audiowrite(['Data/' Name '/sig_y.wav'], sig(:,2), fs);
%     audiowrite(['Data/' Name '/sig_u1.wav'], sig(:,3), fs);
%     audiowrite(['Data/' Name '/sig_u2.wav'], sig(:,4), fs);
end