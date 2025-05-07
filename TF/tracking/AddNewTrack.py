#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 06:27:23 2019

@author: ran
"""
import numpy as np


Addlist=[]
for i in range(0,len(IOUtotal)):
    pp=(IOUtotal[i]!=0).sum(axis=1)==0
    pp=pp.astype(int)
    AddIndex=np.nonzero(pp)
    print(np.array(AddIndex).ndim)
    print('FRAME:',i)
    print(AddIndex)
    
   
    Addlist.append(AddIndex)

#NewT=open(os.path.join(config.otb_data_dir,'RES101/AddNewTrack.txt'), mode= 'w')
#NewT.write()
