import os
import glob
import numpy as np
from scipy.io import wavfile
from scipy.linalg import svd

class Class_XYC_Data_v4:
    def __init__(self, dX=None, dY=None, C=None):
        self.dX = np.array(dX) if dX is not None else np.array([])
        self.dY = np.array(dY) if dY is not None else np.array([])
        self.C = np.array(C) if C is not None else np.array([])

    def LoadMultyfiles_by_Folder_ComPre_ComPost_CounterSt_nRois(self, dir_path, com_pre, com_post, counter_st, n_rois):
        # Mock implementation: Assumes dat files are binary with records of 3 float32 values (dX, dY, C)
        pattern = os.path.join(dir_path, f"{com_pre}*{com_post}*.dat")
        files = glob.glob(pattern)
        dx_all = []
        dy_all = []
        c_all = []
        
        for file_path in files:
            with open(file_path, "rb") as f:
                while True:
                    record = f.read(12)  # 3 float32 values => 4 bytes each
                    if not record:
                        break
                    dX, dY, C = np.frombuffer(record, dtype=np.float32)
                    dx_all.append(dX)
                    dy_all.append(dY)
                    c_all.append(C)
        
        self.dX = np.array(dx_all)
        self.dY = np.array(dy_all)
        self.C = np.array(c_all)

def main():
    data_dir = r'C:\Users\User\Documents\Mafaat_new_Topics\Shirim\DATA'
    dir_lst = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('POS')]
    
    for dir_name in dir_lst:
        dir_path = os.path.join(data_dir, dir_name, 'Patch_1')
        if not glob.glob(os.path.join(dir_path, '*.dat')):
            continue
        
        if 'SPECH' not in dir_name and 'SPEECH' not in dir_name:
            continue
        
        output_dir = os.path.join('Data1', dir_name)
        os.makedirs(output_dir, exist_ok=True)
        
        com_pre = 'CorrelationResult_'
        com_post = '_'
        fs = 8000
        objxyc = Class_XYC_Data_v4()
        objxyc.LoadMultyfiles_by_Folder_ComPre_ComPost_CounterSt_nRois(dir_path, com_pre, com_post, 0, 1)
        
        objxyc.dX[np.isnan(objxyc.dX) | (objxyc.C < 0.7)] = 0
        objxyc.dY[np.isnan(objxyc.dY) | (objxyc.C < 0.7)] = 0
        
        U, S, V = svd(np.vstack([objxyc.dX, objxyc.dY]).T, full_matrices=False)
        sig = np.vstack((objxyc.dX, objxyc.dY, U[:,0],U[:,1])).T
        sig = sig / np.max(np.abs(sig), axis=0)
        
        # Choose a single channel or mix them as needed
        mixed_sig = np.mean(sig, axis=1)
        wavfile.write(os.path.join(output_dir, 'sig.wav'), fs, mixed_sig.astype(np.float32))

if __name__ == '__main__':
    main()
