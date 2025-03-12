import torch
import torchvision.transforms as vis_trans
from PIL import Image
import json
import numpy as np

from mainF.dp import max_index


def imgToTensor(imgSize):
    transList = [
            vis_trans.Resize(imgSize),
            #  vis_trans.RandomCrop(picSize,pad_if_needed=True),
            #  vis_trans.RandomHorizontalFlip(),
            #  vis_trans.RandomVerticalFlip(),
             vis_trans.ToTensor()]
    transfm = vis_trans.Compose(transList)
    return transfm

def tensorToImg(imgSize):
    transList = [
            vis_trans.Resize(imgSize),
            #  vis_trans.RandomCrop(picSize,pad_if_needed=True),
            #  vis_trans.RandomHorizontalFlip(),
            #  vis_trans.RandomVerticalFlip(),
             vis_trans.ToPILImage()]
    transfm = vis_trans.Compose(transList)
    return transfm


class MyImgDataClass():
    def __init__(self, oriImg, device):
        self.device=device
        self.root="dataset/"
        tempDict={}
        tempDict["origin"]=oriImg
        tempDict["mask"]="mask_"+oriImg
        tempDict["normal"]="normal_"+oriImg
        tempDict["texture"]="texture_"+oriImg
        tempDict["result"]="res_"+oriImg
        self.imgDict=tempDict
        
        ori=self.getImg("origin")
        (self.w,self.h)=ori.size

        maskTensor=self.getImgTensor("mask")
        self.maskFlag=((maskTensor>0)[0,0,:])

    def getImg(self, imgName):
        imgPath=self.root+self.imgDict[imgName]
        img=Image.open(imgPath)      
        return img
    
    def getImgTensor(self, imgName, h=-1, w=-1):
        if(h<0 or w <0):
            h=self.h
            w=self.w

        img=self.getImg(imgName)
        imgTrans=imgToTensor((h,w))
        imgTensor=imgTrans(img)
        return imgTensor[None,].to(self.device)#shape:(batch:1, dim, h, w)
    def byMask(self, img, ifMain=True):
        if ifMain:
            img[:,:,~self.maskFlag]=0
        else:
            img[:,:,self.maskFlag]=0
        return img
    
    def initUV(self, ):
        raw_data=torch.randn(size=(1, 2, self.h, self.w))
        # 将数据缩放到0到1范围内
        min_val = raw_data.min()
        max_val = raw_data.max()
        scaled_data = (raw_data - min_val) / (max_val - min_val)
        scaled_data=scaled_data.to(self.device)

        uvByMask=self.byMask(scaled_data)
        return uvByMask

    def getPreUV(self,):
        with open('normal_reUV-main/funcs/img1Json.json', 'r', encoding='utf-8') as f:
            json_str = f.read()
        data = json.loads(json_str)
        data = json.loads(data)
        scores = data["scores"]
        max_index=np.argmax(scores)
        uv_data = data["pred_densepose"][max_index]['uv']
        uv_tensor = torch.tensor(uv_data).float()
        preUv=uv_tensor
        return preUv




    
    def uvReplace(self, newUV):#根据 UV 映射从纹理图像中采样像素，替换原始图像中的对应区域。
        minSize=min(self.h,self.w)

        texture=self.getImgTensor("texture", minSize, minSize)

        min_val = newUV.min()
        max_val = newUV.max()
        newUV_scaled = (newUV - min_val) / (max_val - min_val)*(minSize-1)

        u=newUV_scaled[0,0,:].long()
        v=newUV_scaled[0,1,:].long()

        ori=self.getImgTensor("origin")

        after=ori.clone().detach()
        after[0,:]=texture[0, :, u, v]
        
        res=self.byMask(after)+self.byMask(ori, False)
        self.saveResImg(res)

    def saveResImg(self, imgTensor):
        tensorTrans=tensorToImg((self.h,self.w))
        img=tensorTrans(imgTensor[0])
        savePath=self.root+self.imgDict["result"]
        img.save(savePath)

               
