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
        default=8,
        help="num_workers"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.5
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=1
    )
    parser.add_argument(
        "--model_from",
        type=str,
        default="None.pth"
    )
    # print(parser.parse_args())
    return parser.parse_args()

def modelLoad(fileSave):
    # model=ResNet_main(input_channels=3, tagNum=tagNum).to(device)
    model=UNet(2, True)
    baseEpoch=0
    if Path(fileSave).is_file():
        saveState=torch.load(fileSave, weights_only=True)
        model.load_state_dict(saveState['model_state'])
        baseEpoch=saveState["epoch"]
        print("Model_Load! BaseEpoch:",baseEpoch)
    return model, baseEpoch

def main():
    args=get_args()
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    imgDataset=dataPr.MyImgDataClass(args.oriImg, args.textureImg, device)

    model, baseEpoch=modelLoad(imgDataset.root+args.model_from)
    model=model.to(device)    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

    resUV=train.train_main(imgDataset, model, optimizer, args.epoch, baseEpoch)
    imgDataset.uvReplace(resUV)


if __name__=="__main__":
    main()