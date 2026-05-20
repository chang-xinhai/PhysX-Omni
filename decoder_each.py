import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
import sys
import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
import ipdb
import numpy as np
import torch
import trimesh
# Load a pipeline from a model folder or a Hugging Face model hub.
import re
import gc
def get_sorted_npy_list(folder_path):
    files = os.listdir(folder_path)
    
    pattern = re.compile(r'ind_(\d+)\.npy')
    
    npy_files = []
    for f in files:
        match = pattern.match(f)
        if match:
            npy_files.append((int(match.group(1)), f))
    

    npy_files.sort(key=lambda x: x[0])
    
    sorted_list = [f[1] for f in npy_files]
    count = len(sorted_list)
    
    return count, sorted_list


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default='demo')
parser.add_argument("--imgpath", type=str, default='render')
parser.add_argument("--basepath", type=str, default='demo')
args = parser.parse_args()

pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()




name=args.name
imgpath=args.imgpath
basepath=args.basepath

image = Image.open(imgpath)

qwenpath=os.path.join(basepath,name)
objspath=os.path.join(basepath,name,'objs')


if os.path.exists(os.path.join(qwenpath,'allind.npy')):
    coords=np.load(os.path.join(qwenpath,'allind.npy'))
    eachcoords=[]
    n, sorted_files = get_sorted_npy_list(qwenpath)
    for indname in sorted_files:
        if len(eachcoords)==0:
            slices=[0]
        else:
            slices=[eachcoords[-1][1]]

        slices.append(slices[0]+np.load(os.path.join(qwenpath,indname)).shape[0])
        eachcoords.append(slices)




    size=64
    resolution=64
    
    coords=np.concatenate([np.zeros((len(coords),1)),coords],1)
    coords=torch.Tensor(coords).cuda().int()
    

    
    outputs = pipeline.run_decoder(coords,image,seed=1,eachcoords=eachcoords)
    for i in range(len(outputs)):
        objfilepath=os.path.join(objspath,str(i))
        os.makedirs(os.path.join(objfilepath), exist_ok=True)
        if outputs[i]!=None and len(outputs[i]['mesh'][0].vertices)!=0:

            glb = postprocessing_utils.to_glb(
                outputs[i]['gaussian'][0],
                outputs[i]['mesh'][0],
                # Optional parameters
                simplify=0.5,          # Ratio of triangles to remove in the simplification process
                texture_size=1024,      # Size of the texture used for the GLB
            )

            glb.export(os.path.join(objfilepath,str(i)+'.glb'))

            R = trimesh.transformations.rotation_matrix(np.deg2rad(90), [1, 0, 0])
            glb.apply_transform(R)
            glb.export(os.path.join(objfilepath,str(i)+'.obj'))




