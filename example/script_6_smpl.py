import os

import torch

from creadto._external.smpl.smpl import SMPL


def main():
    smpl_root = "../_external/smpl/"
    neutral_path = os.path.join(smpl_root, "SMPLX_NEUTRAL.pkl")
    neutral = SMPL(path=neutral_path)
    shape = torch.zeros((1, 400), dtype=torch.double)
    poses = torch.zeros((1, 55, 3), dtype=torch.double)
    offset = torch.zeros((1, 3), dtype=torch.double)
    v, _ = neutral(shape, poses, offset)
    n_vertex = v.shape[1]
    pass


if __name__ == "__main__":
    main()
