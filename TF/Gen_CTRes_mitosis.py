import os
import numpy as np



def Gen_CTRes_Mito(file,start,length):
    fn = np.loadtxt(file)
    pad1 = np.zeros([start,4])
    fnlength=len(fn)
    pad2length=length-fnlength-start
    pad2 = np.zeros([pad2length,4])
    padded = np.vstack((pad1,fn,pad2))
    padded.astype(int)

    #rename old file into file_ori
    (filepath, tempfilename) = os.path.split(file)
    (filename, extension) = os.path.splitext(tempfilename)
    name_ori =filepath+'/'+filename+'_ori'+extension
    os.rename(file,name_ori)
    print('Renamed CTRes file')

    #save_new_file
    np.savetxt(file,padded.astype(int), fmt='%i')
    print('Generated new CTRes file')


    return padded

if __name__ == '__main__':
    path_head = '/home/ran/Trails/BF-C2DL-MuSC/CODE/02/CTRes'
    fileidx = str(5)
    path_tail='.txt'
    file = "%s%s%s"%(path_head,fileidx,path_tail)
    padded = Gen_CTRes_Mito(file,15,150)



