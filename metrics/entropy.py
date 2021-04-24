'''
BSD 3-Clause License
Copyright (c) 2017, Tamojit Saha All rights reserved.
Redistribution and use in source and binary forms, with or without modification, 
are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, 
this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, 
this list of conditions and the following disclaimer in the documentation 
and/or other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its contributors 
may be used to endorse or promote products derived from this software without 
specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE 
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import sys

def entropy(signal):
        '''
        function returns entropy of a signal
        signal must be a 1-D numpy array
        '''
        lensig=signal.size
        symset=list(set(signal))
        numsym=len(symset)
        propab=[np.size(signal[signal==i])/(1.0*lensig) for i in symset]
        ent=np.sum([p*np.log2(1.0/p) for p in propab])
        return ent
'''
#..........................Sample Usage.................................#
Copy this....
imageEntropy('lena.tiff', "Entropy in 10 x 10 neighbourhood", 10, 8, 600)
#.......................................................................#
'''   
    
def imageEntropy(file_name, image_caption, image_height, image_width, DPI):
    colorIm=Image.open(file_name)
    base = os.path.basename(file_name)
    name = os.path.splitext(base)[0]
    ext = os.path.splitext(base)[1]
    greyIm=colorIm.convert('L')
    colorIm=np.array(colorIm)
    greyIm=np.array(greyIm)
    N=5
    S=greyIm.shape
    E=np.array(greyIm)
    for row in range(S[0]):
            for col in range(S[1]):
                    Lx=np.max([0,col-N])
                    Ux=np.min([S[1],col+N])
                    Ly=np.max([0,row-N])
                    Uy=np.min([S[0],row+N])
                    region=greyIm[Ly:Uy,Lx:Ux].flatten()
                    E[row,col]=entropy(region)

    plt.subplot(1,3,3)
    plt.imshow(E, cmap=plt.cm.jet)
    plt.xlabel(image_caption)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(image_height, image_width)#in inches
    # when saving, specify the DPI
    new_file_name = "entropy_"+name+ext
    print ("\nThe filename is: "+new_file_name)
    plt.savefig(new_file_name, dpi = DPI, bbox_inches='tight')
    #plt.colorbar()        
    # plt.show()


imageEntropy(sys.argv[1], "Entropy "+sys.argv[1][:-4], 10, 8, 600)