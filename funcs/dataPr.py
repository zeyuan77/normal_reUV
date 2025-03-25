import torch
import torchvision.transforms as vis_trans
from PIL import Image
import json
import numpy as np
        

def imgToTensor(imgSize=None):
    transList = [
            # vis_trans.Resize(imgSize),
            #  vis_trans.RandomCrop(picSize,pad_if_needed=True),
            #  vis_trans.RandomHorizontalFlip(),
            #  vis_trans.RandomVerticalFlip(),
             vis_trans.ToTensor()]
    transfm = vis_trans.Compose(transList)
    return transfm


def tensorToImg(imgSize=None):
    transList = [
            # vis_trans.Resize(imgSize),
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
    def __init__(self, oriName, textureName, device):
        self.device=device
        self.root="dataset/"

        self.oriName=oriName
        self.resImg=f"{oriName}_by_{textureName}.jpg"

        tempDict={}
        tempDict["origin"]=f"{oriName}.jpg"
        tempDict["mask"]=f"mask_{oriName}.jpg"
        tempDict["normal"]=f"normal_{oriName}.jpg"
        tempDict["texture"]=f"{textureName}.jpg"
        self.imgDict=tempDict
        
        self.cropBox, self.preUV=self.getDenseposeData()
        
        self.oriSize, ori_crop=self.getImg("origin")
        self.cropSize=ori_crop.size

        maskTensor=self.getImgTensor("mask", self.oriSize)
        self.maskFlag=((maskTensor>0)[0,0,:])#shape:(h,w)


    def getDenseposeData(self,):
        #from densepose
        jsonName=f"preUV_{self.oriName}.json"
        jsonnPath=self.root+jsonName

        with open(jsonnPath, 'r', encoding='utf-8') as f_densepose:
            json_str = json.load(f_densepose)#str
        denseposeDict= json.loads(json_str)#dict

        cropBox=denseposeDict['pred_boxes_XYXY'][0]
        cropBox_int=[int(i) for i in cropBox]
        # print(cropBox, cropBox_int)

        scores = denseposeDict["scores"]
        max_idx=np.argmax(scores) 
        preUV = denseposeDict["pred_densepose"][max_idx]['uv']

        return cropBox_int, preUV
    
    
    def getImg(self, imgName, re_wh=(0, 0), ifCrop=True):
        imgPath=self.root+self.imgDict[imgName]
        img=Image.open(imgPath)  
        oriSize=img.size

        if re_wh[0]*re_wh[1]>0:
            img=img.resize(re_wh, resample=2)
        if ifCrop:
            img=img.crop(self.cropBox)    
        return oriSize, img
    
    
    def getImgTensor(self, imgName, re_wh=(0, 0), ifCrop=True):
        _, img=self.getImg(imgName, re_wh, ifCrop)
        imgTrans=imgToTensor()
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

    
    def getPreUV_Mask(self, ):
        # preUV=self.initUV()
        #from densepose
        preUV_tensor= torch.tensor(self.preUV)[None,]#shape:(batch:1, uv:2, h, w)

        preUV_byMask=self.byMask(preUV_tensor)
        return preUV_byMask.to(self.device)
    
    
    def saveModel(self, epoch, model):
        modelName=f"epoch{epoch}_{self.oriName}.pth"
        savePath = self.root + modelName
        torch.save({"epoch": epoch,
                    "model_state": model.state_dict()
                    }, savePath)
        
    
    def uvReplace(self, newUV):
        minSize=min(self.cropSize)
        texture=self.getImgTensor("texture", re_wh=(minSize, minSize), ifCrop=False)
        newUV_scaled=dataN(newUV)*(minSize-1)

        u=newUV_scaled[0,0,:].long()
        v=newUV_scaled[0,1,:].long()

        ori=self.getImgTensor("origin")

        after=ori.clone().detach()
        after[0,:]=texture[0, :, u, v]
        
        res=self.byMask(after)+self.byMask(ori, False)
        self.saveResImg(res)


    def saveResImg(self, imgTensor):
        tensorTrans=tensorToImg()
        img=tensorTrans(imgTensor[0])
        img.save(self.root+self.resImg)



               
