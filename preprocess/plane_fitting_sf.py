import torch
import tqdm
import numpy as np
import HitnetModule
from pathlib import Path
import multiprocessing as mp
import imp
 
# Specify the path to the gfg.py module
module_path = '/media/rvcse22/CSERV/cse/TinyHITNet-master/dataset/utils.py'
 
# Load the module using imp.load_source
ds_ut= imp.load_source('ds_ut', module_path)
 
# Now you can use the functions or variables def
readPFM=ds_ut.readPFM
np2torch=ds_ut.np2torch
#from dataset.utils import readPFM, np2torch


def process(file_path):
    while True:
        for ids, lock in enumerate(process.lock_list):
            if lock.acquire(block=False):
                pfm_path = (process.root / "disparity" / "disparity" / file_path).with_suffix(".pfm")
                dxy_path = (process.root / "slant_window" / file_path).with_suffix(
                    ".npy"
                )
                dxy_path.parent.mkdir(exist_ok=True, parents=True)
                with torch.no_grad():
                    x = np2torch(readPFM(pfm_path)).unsqueeze(0).cuda(0)
                    x = HitnetModule.plane_fitting(x, 256, 0.1, 9, 0, 1e5)
                    x = x[0].cpu().numpy()
                np.save(dxy_path, x)
                lock.release()
                return


def process_init(lock_list, root):
    process.lock_list = lock_list
    process.root = root


def main(root, list_path):
    root = Path(root)

    with open(list_path, "rt") as fp:
        file_list = [Path(line.strip()) for line in fp]

    lock_list = [mp.Lock() for _ in range(8)]
    with mp.Pool(8, process_init, [lock_list, root]) as pool:
        list(tqdm.tqdm(pool.imap_unordered(process, file_list), total=len(file_list)))


if __name__ == "__main__":
    main("/media/rvcse22/CSERV/cse/stereo_depth_datasets/MAINDS_FLY_TH_3D", "lists/sceneflow_train_fly3d_only.list")
    main("/media/rvcse22/CSERV/cse/stereo_depth_datasets/MAINDS_FLY_TH_3D", "lists/sceneflow_test.list")
