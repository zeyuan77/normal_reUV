import torch
import argparse

from func_import import *
from models import UNet

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("oriImg", type=str)

    parser.add_argument("textureImg", type=str)

    parser.add_argument(
        "--numWk",
        type=int,
        default=4,
        help="num_workers"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.05
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=37
    )
    # print(parser.parse_args())
    return parser.parse_args()

def main():
    args=get_args()
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    imgDataset=dataPr.MyImgDataClass(args.oriImg, args.textureImg, device)

    model=UNet(2, True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

    resUV=train.train_main(imgDataset, model, optimizer, args.epoch)
    imgDataset.uvReplace(resUV)


if __name__=="__main__":
    main()