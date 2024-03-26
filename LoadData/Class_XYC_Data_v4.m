classdef Class_XYC_Data_v4


    properties
        dX = [] ;
        dY = [] ;
        C = [] ;
        fps = 1;
        TimeVector = [] ;

    end

    properties  (Dependent)
        Mess
    end


    properties (SetAccess=protected)
        LoadFiesList = [] ;
        GPSTime_ms = [] ;
    end


    methods % main

        function obj = Class_XYC_Data_v4(XList,YList,CList)
            if nargin>1
                obj.dX = XList ; %XList(time,sRoi)
                obj.dY = YList ; %XList(time,sRoi)
                obj.C  = CList ; %XList(time,sRoi)
            end
            [~] = obj.CheckValidity  ;

        end



        function objPlus = plus(obj1,obj2)

            P1 = isa(obj1,"Class_XYC_Data_v4") ;
            P2 = isa(obj2,"Class_XYC_Data_v4") ;

            if (P1 && P2)
                dx = [ obj1.dX ; obj2.dX ] ;
                dy = [ obj1.dY ; obj2.dY ] ;
                Cv = [ obj1.C  ; obj2.C  ] ;

                objPlus = Class_XYC_Data_v4(dx,dy,Cv) ;
                objPlus.fps = obj1.fps ;
                objPlus.TimeVector = [obj1.TimeVector , obj2.TimeVector] ;
                objPlus.GPSTime_ms = [obj1.GPSTime_ms , obj2.GPSTime_ms] ;
                objPlus.LoadFiesList = [obj1.LoadFiesList ; obj2.LoadFiesList] ;
            else
                beep() ;
                disp(['   '] )
                disp(['    "+"  must be used between two "Class_XYC_Data_v4" objects  !!!'] )
                disp(['   '] )
                objPlus = [] ;
            end



        end




        function obj = BuildGpsTimeVec_by_Onefilename(obj,filename,nRois)
            fpsV  =  obj.fps ;
            fileObj  = Class_File_CorrelationResults_4(filename,nRois) ;
            timems = fileObj.get_GPSTime_byFilename(fpsV) ;
            obj.GPSTime_ms = timems ;
        end


        function obj = BuildGpsTimeVec_by_AllLoadFiesList(obj,nRois)
            timeAll = []     ;
            Stind =  2       ;
            fpsV  =  obj.fps ;
            NumFiles = length(obj.LoadFiesList) ;

            if NumFiles < Stind
                Stind = 1 ;
            end

            UpToFrame  = 0 ;
            for ii = 1:Stind
                filename = obj.LoadFiesList{ii} ;
                fileObj  = Class_File_CorrelationResults_4(filename,nRois) ;
                UpToFrame = UpToFrame + fileObj.fr_inFile ;
            end
            dFrToSt = UpToFrame - fileObj.fr_inFile ; 
            CorrRes_ms = obj.F2_CorrResChar_to_time_ms(filename) ;
            StTime_ms = CorrRes_ms - 1000*(dFrToSt/fpsV) ;
            timeAll = StTime_ms + 1000*((0:(UpToFrame-1))/fpsV) ;

            for kk=(Stind+1):NumFiles
                filename = obj.LoadFiesList{kk} ;
                fileObj  = Class_File_CorrelationResults_4(filename,nRois) ; 
                timems = fileObj.get_GPSTime_byFilename(obj.fps) ;
                timeAll = [timeAll , timems] ;
            end

            obj.GPSTime_ms = timeAll ;

        end



        function obj = LoadData_by_Filefullpath_nRois(obj,fullpathstring,nRois)

            TmpObj = Class_File_CorrelationResults_4(fullpathstring,nRois);
            obj.dX = TmpObj.dx ;
            obj.dY = TmpObj.dy ;
            obj.C = TmpObj.C ;
            obj.LoadFiesList = fullpathstring ;

        end


        function saveToCorrRes_by_DestFullFilePath(obj,FullFilePath)
            MtxX = obj.dX' ;
            MtxY = obj.dY' ;
            MtxC = obj.C' ;
            obj.Fun21b_Mat2CorrRes(MtxX,MtxY,MtxC,FullFilePath) ;
        end

        function obj = LoadMultyfiles_by_Folder_ComPre_ComPost_CounterSt_nRois(obj,Folder,ComPreStr,ComPostStr,CounterSt,nRois)
            ClFolder = Class_Folder_v2(Folder);
            FilesCellList = ClFolder.get_directFiles_byPartialString(ComPreStr,[]) ;
            L = length(FilesCellList) ;
            dyall = [];
            dxall = [];
            Call = [];
            fileFPobj = Class_FullPath(FilesCellList{1});
            fn = fileFPobj.filename ;
            f1 = strfind(fn,ComPreStr) ;
            LenComPreStr = length(ComPreStr) ;
            filePrefixTemp = fn(1:LenComPreStr+f1-1) ;

            count = CounterSt-1;
            nn=1 ;
            for kk = 1:L
                count = count+1;
                tmp = [fileFPobj.location,'\',filePrefixTemp,num2str(count),ComPostStr,'*',fileFPobj.type] ;
                dirRes = dir(tmp)  ;

                if ~isempty(dirRes)
                    FFname = fullfile(dirRes.folder,dirRes.name);
                    tmp = Class_File_CorrelationResults_4(FFname,nRois);
                    dxall = [dxall;tmp.dx] ;
                    dyall = [dyall;tmp.dy] ;
                    Call = [Call;tmp.C] ;
                    tmp=[];
                    FFnameAll{nn} = FFname ;
                    nn=nn+1 ;

                end
            end

            obj.dX = dxall ;
            obj.dY = dyall ;
            obj.C = Call;

            obj.LoadFiesList = FFnameAll' ;

        end

        function obj = LoadMultyfiles_by_Folder_PartStr_nRois(obj,Folder,fileTemp,nRois)
            ClFolder = Class_Folder_v2(Folder);
            FilesCellList = ClFolder.get_directFiles_byPartialString(fileTemp,[]) ;
            L = length(FilesCellList) ;
            dyall = [];
            dxall = [];
            Call = [];

            for kk = 1:L
                tmp = Class_File_CorrelationResults_4(FilesCellList{kk},nRois);
                dxall = [dxall;tmp.dx] ;
                dyall = [dyall;tmp.dy] ;
                Call = [Call;tmp.C] ;
                tmp=[];
            end

            obj.dX = dxall ;
            obj.dY = dyall ;
            obj.C = Call ;

            obj.LoadFiesList = FilesCellList' ;

        end




        function Spect = getSpectrogram_SpotIndx_Winsec_XYC(obj,SpotIndx,Winsec,XYC)

            vecdata = obj.get_Specific_SpotData(XYC,SpotIndx) ;
            ShiftObj = Class_Shift_1Data(vecdata,obj.fps) ;
            Spect = ShiftObj.getSpectrogram_Winsec(Winsec) ;

        end

        function MyDisplay_SpotIndx_XYC(obj,SpotIndx,XYC)

            vecdata = obj.get_Specific_SpotData(XYC,SpotIndx) ;
            ShiftObj = Class_Shift_1Data(vecdata,obj.fps) ;
            ShiftObj.MyDisplay;

            ch = get(gcf,'Children') ;
            if XYC == 'C'
                ch(4).Title.String = strrep( ch(4).Title.String,'Shift','CorrVal');
            else
                ch(4).Title.String = strrep( ch(4).Title.String,'Shift',['d',XYC]);
            end


        end


        function ShiftOut = extractOneShiftObj_SpotIndx_XYC(obj,SpotIndx,XYC)

            vecdata = obj.get_Specific_SpotData(XYC,SpotIndx) ;
            ShiftOut = Class_Shift_1Data(vecdata,obj.fps) ;

        end


    end % Main

    methods %set

    end

    methods %get

        function Mess = get.Mess(obj)

            Valid = CheckValidity(obj) ;
            switch Valid
                case 1
                    Mess = 'Ok' ;
                case 0
                    Mess = ' Data fields are not equal size !!' ;
            end

        end

    end


    methods (Access=protected,Static)
        %%%%%
        function Fun21b_Mat2CorrRes(MtxX,MtxY,MtxC,savetto)
            %  MtxX(sRoi,xshift)  ;
            %  MtxY(sRoi,yshift)  ;
            %  MtxC(sRoi,corrval )  ;

            M = zeros(3*length(MtxX(:)),1);
            M(1:3:end)  = MtxX(:) ;
            M(2:3:end)  = MtxY(:) ;
            M(3:3:end)  = MtxC(:) ;

            % savetto = fullfile(DestFolder,filename);
            [fb]= fopen(savetto,'wb') ;
            fwrite(fb,M,'float32') ;
            fclose(fb);


        end
        %%%%%
        function CorrRes_ms = F2_CorrResChar_to_time_ms(CorrReschar)
            fromEnd = 4;
            ff = find(CorrReschar=='_');
            Stpoint=ff(end)+1;
            CorrRes_ms = str2double(CorrReschar(Stpoint:end-fromEnd));
        end

    end

    methods (Access=protected)

        function vecdata = get_Specific_SpotData(obj,XYC,sRoi)
            switch  XYC
                case 'Y'
                    vecdata = squeeze(obj.dY(:,sRoi)) ;
                case 'X'
                    vecdata = squeeze(obj.dX(:,sRoi)) ;
                case 'C'
                    vecdata = squeeze(obj.C(:,sRoi)) ;
            end

        end

        function Valid = CheckValidity(obj)
            Sx = size(obj.dX) ;
            Sy = size(obj.dY) ;
            Sc = size(obj.C) ;
            CheckSameSizeA = (sum(Sx==Sy)==2) ;
            CheckSameSizeB = (sum(Sy==Sc)==2) ;

            if CheckSameSizeA==1 && CheckSameSizeB==1
                Valid = 1 ;
            else

                Valid = 0 ;

            end

        end

    end


end %Class
