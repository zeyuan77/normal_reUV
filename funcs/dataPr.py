import torch
import torchvision.transforms as vis_trans
from PIL import Image
import json
        
def imgToTensor(imgSize):#将图像转化为Pytorch张量
    transList = [
            vis_trans.Resize(imgSize),#调整大小
            #  vis_trans.RandomCrop(picSize,pad_if_needed=True),
            #  vis_trans.RandomHorizontalFlip(),
            #  vis_trans.RandomVerticalFlip(),
             vis_trans.ToTensor()]
    transfm = vis_trans.Compose(transList)
    return transfm

def tensorToImg(imgSize):#将Pytorch张量转换回图像
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
        self.device=device#设备
        self.root="dataset/"#根目录
        tempDict={}#字典
        tempDict["origin"]=oriImg
        tempDict["mask"]="mask_"+oriImg
        tempDict["normal"]="normal_"+oriImg
        tempDict["texture"]="texture_"+oriImg
        tempDict["result"]="res_"+oriImg
        self.imgDict=tempDict
        
        ori=self.getImg("origin")#加载原始图像
        (self.w,self.h)=ori.size#原始尺寸

        maskTensor=self.getImgTensor("mask")#加载掩码图像
        self.maskFlag=((maskTensor>0)[0,0,:])#shape:(h,w) 生成布尔掩码

    def getImg(self, imgName):
        imgPath=self.root+self.imgDict[imgName]#拼接图像路径
        img=Image.open(imgPath)      
        return img
    
    def getImgTensor(self, imgName, h=-1, w=-1):
        if(h<0 or w <0):#未指定 h and w
            h=self.h
            w=self.w

        img=self.getImg(imgName)
        imgTrans=imgToTensor((h,w))
        imgTensor=imgTrans(img)
        return imgTensor[None,].to(self.device)#shape:(batch:1, dim, h, w)
        #将张量扩展到批次维度，并将其移动到指定设备
    def byMask(self, img, ifMain=True):
        if ifMain:
            img[:,:,~self.maskFlag]=0#ifMain为True则保留掩码区域，非掩码区域置零
        else:
            img[:,:,self.maskFlag]=0
        return img
    
    def initUV(self, ):#生成一个随机的UV映射张量
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
        if isinstance(data, str):
            data = json.loads(data)
        uv_data = data["pred_densepose"][0]['uv']
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

               
