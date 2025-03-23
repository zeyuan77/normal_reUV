import torch
import torchvision.transforms as vis_trans
from PIL import Image
import json
import numpy as np
        

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


def dataN(raw):
    # 将数据缩放到0到1范围内
    min_val = raw.min()
    max_val = raw.max()
    scaled = (raw - min_val) / (max_val - min_val)
    return scaled


class MyImgDataClass():
    def __init__(self, oriImg, textureImg, device):
        self.device=device
        self.root="dataset/"

        self.oriImg=oriImg
        self.resImg=f"{oriImg}_by_{textureImg}.jpg"

        tempDict={}
        tempDict["origin"]=f"{oriImg}.jpg"
        tempDict["mask"]=f"mask_{oriImg}.jpg"
        tempDict["normal"]=f"normal_{oriImg}.jpg"

        tempDict["texture"]=f"{textureImg}.jpg"

        self.imgDict=tempDict
        
        ori=self.getImg("origin")
        (self.w,self.h)=ori.size

        maskTensor=self.getImgTensor("mask")
        self.maskFlag=((maskTensor>0)[0,0,:])#shape:(h,w)

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
    
    # def initUV(self, ):#not use
    #     raw_data=torch.randn(size=(1, 2, self.h, self.w))
    #     scaled_data=dataN(raw_data).to(self.device)
    #     uvByMask=self.byMask(scaled_data)
    #     return uvByMask
    
    def getPreUV(self, ):
        # preUV=self.initUV()
        #from densepose
        jsonName=f"preUV_{self.oriImg}.json"
        jsonnPath=self.root+jsonName

        with open(jsonnPath, 'r', encoding='utf-8') as f_densepose:
            json_str = json.load(f_densepose)#str
        denseposeDict= json.loads(json_str)#dict
        
        scores = denseposeDict["scores"]
        max_index=np.argmax(scores)
        
        preUV = denseposeDict["pred_densepose"][max_index]['uv']
        preUV_tensor= torch.tensor(preUV)[None,]#shape:(batch:1, uv:2, h, w)

        preUV_byMask=self.byMask(preUV_tensor)
        return preUV_byMask.to(self.device)
    
    def saveModel(self, epoch, model):
        modelName=f"epoch{epoch}_{self.oriImg}.pth"
        savePath = self.root + modelName
        torch.save({"epoch": epoch,
                    "model_state": model.state_dict()
                    }, savePath)
    
    def uvReplace(self, newUV):
        minSize=min(self.h,self.w)
        texture=self.getImgTensor("texture", minSize, minSize)
        newUV_scaled=dataN(newUV)*(minSize-1)

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
        img.save(self.root+self.resImg)



               
