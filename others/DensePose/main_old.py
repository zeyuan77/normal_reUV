import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image,ImageEnhance

import json

from seg.seg import SegAPI
import cv2
import numpy as np
import os
import shutil
import sys
sys.path.append('./DeepBump')
from DeepBump.cli import getNormal 
from PIL import Image

sys.path.append("PIE")
from PIE.Eval import pie

from skimage.metrics import structural_similarity as ssim


def clear_folder(folder_path):
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)
    except Exception as e:
        print(f"发生错误: {str(e)}")

def recon(alb,shd):
    reflectance_array = np.array(alb)
    shading_array = np.array(shd)
    shading_array = np.stack((shading_array,) * 3, axis=-1)
    result_array= np.multiply(reflectance_array, shading_array/255.0)
    result_array = result_array.astype(np.uint8)
    result_image = Image.fromarray(result_array)
    return result_image

def intrinsic_decomposition(img,w,h):
    visuals = 'PIE/test_outputs/'
    data_root = 'PIE/test_inputs/'

    clear_folder(data_root)
    clear_folder(visuals)
    img.save(os.path.join(data_root, "img.png"))
    pie(w,h)
    alb=Image.open( visuals+"img_pred_alb.png")#albedo
    shd=Image.open( visuals+"img_pred_shd.png")#shading
    return alb,shd

def json2tensor(path):
    with open(path, 'r') as jsonFile:
        data = json.load(jsonFile)
    merged_labels = []
    merged_uv = []
    for result in data:
        if "pred_densepose" in result:
            densepose_data = result["pred_densepose"]
            densepose_tensors = []

            for densepose_dict in densepose_data:
                # 从字典中获取 labels 和 uv 数据
                labels = torch.tensor(densepose_dict["labels"])
                uv = torch.tensor(densepose_dict["uv"])

                merged_labels.append(labels)
                merged_uv.append(uv)
        if "pred_boxes_XYXY" in result:
            xyxy_data = result["pred_boxes_XYXY"]
            xyxy = []
            xyxy.append(xyxy_data[0])
            xyxy.append(xyxy_data[1])
            xyxy.append(xyxy_data[2])
            xyxy.append(xyxy_data[3])
            
    merged_labels_tensor = torch.cat(merged_labels, dim=0)
    merged_uv_tensor = torch.cat(merged_uv, dim=0)
    print(merged_labels_tensor.shape,merged_uv_tensor.shape)
    return merged_labels_tensor,merged_uv_tensor,xyxy

import subprocess
img_path="7.jpg"
tex_path="13.jpg"
command = "python apply_net1.py dump configs/densepose_rcnn_R_50_FPN_s1x.yaml model_final_162be9.pkl "+img_path
# 使用subprocess运行命令
process = subprocess.Popen(command, shell=True)
process.wait() 

label_tensor, uv_tensor, xyxy =json2tensor("result.json")
target_size=(32,32)


#按densepose的结果裁剪img
def preprocess_img(path):
    img=Image.open(path)
    box = (
        xyxy[0],
        xyxy[1],
        xyxy[2],
        xyxy[3]
    )
    cropped_img = img.crop(box)
    cropped_img.save("process_"+path)
    return "process_"+path


class CNN(nn.Module):
    def __init__(self, input_channels, output_size,h,w):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128* h * w, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
    def model_save(self):
            torch.save(self.state_dict(), 'weights/model_weights.pkl')


class UVPredictor:
    def __init__(self,path,texture_path,densepose_tensor):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("GPU 可用")
        else:
            self.device = torch.device("cpu")
            print("GPU 不可用")
        preprocess_img_path=preprocess_img(path)
        img=Image.open(preprocess_img_path)
        self.img_w,self.img_h=img.size
        self.img=img
        
        self.alb,self.shd=intrinsic_decomposition(img,512,512)

        enhancer = ImageEnhance.Contrast(self.shd)
        factor = 1
        self.shd = enhancer.enhance(factor)
        self.alb=self.alb.resize((self.img_w,self.img_h), Image.BILINEAR) #alb变为原图大小
        self.shd=self.shd.resize((self.img_w,self.img_h), Image.BILINEAR)
        
        # self.mask=get_mask(preprocess_img_path).resize(target_size)
        self.mask=Image.open("mask_"+preprocess_img_path)
        self.texture=Image.open(texture_path)

        self.preprocess = transforms.Compose([transforms.ToTensor()])
        #self.normal=((self.preprocess(getNormal(Image.open(preprocess_img_path)).resize(target_size)) -0.5)*2).to(self.device)
        self.normal=self.preprocess(Image.open("normal_"+path)).to(self.device)
        print("normal_"+path)
        #print(self.normal)
        # print(self.normal)
        # print((self.normal[0, y, x] / self.normal[2, y, x])**2, (self.normal[1,y,x]/self.normal[2,y,x])**2,(self.normal[0,y,x]*self.normal[1,y,x])/(self.normal[2,y,x])**2 )
        #self.normal=(self.normal - 0.5) * 2
        # print(self.normal)
        # print((self.normal[0, y, x] / self.normal[2, y, x])**2, (self.normal[1,y,x]/self.normal[2,y,x])**2,(self.normal[0,y,x]*self.normal[1,y,x])/(self.normal[2,y,x])**2 )
        #self.normal=self.preprocess(Image.open("normal_5.jpg").resize(target_size)).to(self.device)

        print("normal tensor shape:",self.normal.shape)
        self.dense_pose=densepose_tensor.to(self.device)
        print("densepose tensor shape:",self.dense_pose.shape)
        # self.normal__tensor=self.get_tensor(self.img,self.mask)
        # self.dense_pose_tensor=self.get_tensor(self.img,self.mask)    #这里之后要改
        self.uv_tensor=self.dense_pose
        self.preprocess_blocks(128,32)

    def preprocess_blocks(self, block_size, overlap):
        stride = block_size - overlap
        self.block_size=block_size
        self.overlap=overlap

        self.img_blocks = []
        self.mask_blocks = []
        self.alb_blocks,self.shd_blocks=[],[]
        self.uv_blocks,self.densepose_block=[],[]
        self.normal_blocks=[]
        self.coordinates_tensor=[]

        print(self.uv_tensor.shape)
        for y in range(0, self.img_h, stride):
            for x in range(0, self.img_w, stride):
                if(self.img_w-x<block_size):
                    x_step=self.img_w-x
                else:
                    x_step=block_size
                if(self.img_h-y<block_size):
                    y_step=self.img_h-y
                else:
                    y_step=block_size
                block_img = self.img.crop((x, y, x + x_step, y + y_step))
                block_mask = self.mask.crop((x, y, x +x_step, y + y_step))
                block_alb=self.alb.crop((x, y, x +x_step, y + y_step))
                block_shd=self.shd.crop((x, y, x +x_step, y + y_step))
                block_uv = self.uv_tensor[:, y:y + y_step, x: x + x_step]
                block_normal=self.normal[:, y:y + y_step, x: x + x_step]


                self.img_blocks.append(block_img)
                self.mask_blocks.append(block_mask)
                self.alb_blocks.append(block_alb)
                self.shd_blocks.append(block_shd)
                tensor=F.interpolate(block_uv.unsqueeze(0) ,size=target_size, mode='bilinear', align_corners=False).to(self.device)
                self.densepose_block.append(tensor.squeeze(0))
                self.uv_blocks.append(tensor)
                # print(tensor)
                # print(tensor.shape)
                self.normal_blocks.append(F.interpolate(block_normal.unsqueeze(0) ,size=target_size, mode='bilinear', align_corners=False).squeeze(0).to(self.device))

        for mask_item in self.mask_blocks:
            mask_tensor = self.preprocess(mask_item)
            non_zero_indices = torch.nonzero(mask_tensor)
            # x_coords =torch.round( non_zero_indices[:, 2]/self.block_size*target_size[0]).int()
            # y_coords = torch.round(non_zero_indices[:, 1]/self.block_size*target_size[1]).int()
            x_coords = non_zero_indices[:, 2]
            y_coords = non_zero_indices[:, 1]
            coordinates=torch.cat((x_coords.unsqueeze(1), y_coords.unsqueeze(1)), dim=1).to(self.device)
            rounded_tensor = torch.round(coordinates /self.block_size*(target_size[0]-1)).long()
            unique_tensor = torch.unique(rounded_tensor, dim=0)
            self.coordinates_tensor.append(unique_tensor)



    def combine_blocks(self):
        stride = self.block_size - self.overlap
        combined_img = Image.new('RGB', (self.img_w, self.img_h))
        combined_mask = Image.new('L', (self.img_w, self.img_h))

        block_index = 0
        for y in range(0, self.img_h, stride):
            for x in range(0, self.img_w, stride):
                block_img = self.alb_blocks[block_index]
                print(block_img.size)
                block_mask = self.mask_blocks[block_index]

                combined_img.paste(block_img, (x, y))
                combined_mask.paste(block_mask, (x, y))

                block_index += 1

        combined_mask.save("combined_7_mask.jpg")
        combined_img.save("combined_7_alb.jpg")




    def criterion(self, uv_tensor,epoch,index):
        Lambda1 = 1
        Lambda2 = 0.2
        Lambda3 = 0.01
        w,h= target_size[0],target_size[1]
        
        mask = (self.coordinates_tensor[index][:, 0] == 0) | (self.coordinates_tensor[index][:, 0] == w - 1) | (self.coordinates_tensor[index][:, 1] == 0) | (self.coordinates_tensor[index][:, 1] == h - 1)
        x = self.coordinates_tensor[index][:, 0][~mask]
        y = self.coordinates_tensor[index][:, 1][~mask]   

        du_x = uv_tensor[0, y*w+ x + 1]*128 - uv_tensor[0, y*w+ x]*128
        dv_x = uv_tensor[0,h*w+ y*w+ x + 1]*128 - uv_tensor[0,h*w+ y*w+ x]*128

        du_y = uv_tensor[0, (y+1)*w+ x ]*128 - uv_tensor[0, y*w+ x]*128
        dv_y = uv_tensor[0,h*w+ (y+1)*w+ x ] *128- uv_tensor[0,h*w+ y*w+ x]*128
        
        normal=self.normal_blocks[index]
        normal_diff_xz = (normal[0, y, x] / normal[2, y, x])**2
        normal_diff_yz = (normal[1,y,x]/normal[2,y,x])**2
        normal_diff_xyz=(normal[0,y,x]*normal[1,y,x])/(normal[2,y,x])**2 
        dense_pose_diff = (uv_tensor[0,x+ y*w ]*128 - self.densepose_block[index][0, y, x]*128)**2 + (uv_tensor[0,w*h+y*w+x]*128 - self.densepose_block[index][1, y, x]*128)**2
        
        term1 = Lambda1 *( (du_x**2+dv_x**2-1-normal_diff_xz)**2
                          +(du_y**2+dv_y**2-1-normal_diff_yz)**2
                           + (du_x*du_y+dv_x*dv_y-normal_diff_xyz)**2)  
        term2 = Lambda2 * dense_pose_diff
        
        term3 = Lambda3 * torch.clamp(du_x * dv_y - du_y*dv_x, min=0)
        if(epoch%100==0):
            print("term1:",torch.sum(term1))
            print("term2:",torch.sum(term2))
            print("term3:",torch.sum(term3))
        loss = torch.sum(term1 + term2 + term3)

        return loss

    def predict(self):
        texture_pixels=self.texture.load()
        out_imgs=[]
        block_size=(self.block_size,self.block_size)
        for index, value in enumerate(self.img_blocks):
            input_channels = 2
            output_size = target_size[0] * target_size[1] * 2

        #     model = CNN(input_channels, output_size,target_size[1],target_size[0])
        #     #model.load_state_dict(torch.load('weights/model_weights.pkl'))
        #     model.to(self.device)
        #     optimizer = optim.Adam(model.parameters(), lr=0.001)
            
        # # uv预测
        #     num_epochs =1
        #     last_loss=0
        #     for epoch in range(num_epochs):
        #         outputs = model(self.uv_blocks[index])
        #         loss = self.criterion(outputs,epoch,index)
        #         if loss!=0:
        #             print(outputs)

        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #         if(epoch%100==0):
        #             if(abs(loss-last_loss)/last_loss<0.01 or loss<1000):
        #                 break
        #             last_loss=loss
        #             print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

        #     outputs.to("cpu")
            
            output_img=self.alb_blocks[index].copy().resize(block_size)
            output_img_pixels = output_img.load()

            mask_tensor = self.preprocess(self.mask_blocks[index].resize(block_size))
            non_zero_indices = torch.nonzero(mask_tensor)
            x_coords = non_zero_indices[:, 2]
            y_coords = non_zero_indices[:, 1]
            coordinates_tensor = torch.cat((x_coords.unsqueeze(1), y_coords.unsqueeze(1)), dim=1).to("cpu")
            num_rows, _ = coordinates_tensor.shape


            outputs=self.uv_blocks[index].view(1,2, target_size[0], target_size[0])
            upsampled_tensor = F.interpolate(outputs, size=(block_size), mode='bilinear', align_corners=False)
            outputs=upsampled_tensor.view(1, 2*self.block_size*self.block_size)

            w,h=self.texture.size
            for i in range(num_rows):
                x,y=coordinates_tensor[i]
                output_img_pixels[x,y]=texture_pixels[torch.clamp(outputs[0, y*self.block_size+ x ],0,0.99)*w,torch.clamp(outputs[0,self.block_size*self.block_size+ y*self.block_size+ x ],0,0.99)*h]
                #output_img_pixels[x,y]=texture_pixels[torch.clamp(outputs[0, y*self.img_w+ x ],0,0.99)*w,torch.clamp(outputs[0,self.img_w*self.img_h+ y*self.img_w+ x ],0,0.99)*h]
            output_img=recon(output_img, self.shd_blocks[index].resize(block_size))
            output_img.resize(value.size)
            #output_img.save("tile_"+str(index)+img_path)
            out_imgs.append(output_img)

        stride = self.block_size - self.overlap
        combined_img = Image.new('RGB', (self.img_w, self.img_h))
        block_index = 0
        for y in range(0, self.img_h, stride):
            for x in range(0, self.img_w, stride):
                block_img = out_imgs[block_index]
                combined_img.paste(block_img, (x, y))
                block_index += 1
        combined_img.save("mlp"+img_path)


w,h=Image.open("process_7.jpg").size
tensor = torch.zeros(2, h,w)
for x in range(w):
    for y in range(h):
        tensor[0, y,x] = y / h
        tensor[1, y,x] = x / w

# stride = 128-32
# combined_img = Image.new('RGB', (w,h))
# block_index = 0
# for y in range(0, h, stride):
#     for x in range(0, w, stride):
#         block_img = Image.open("tile_"+str(block_index)+"7.jpg")
#         combined_img.paste(block_img, (x, y))
#         block_index += 1
# combined_img.save("combined_7_1.jpg")


#img=Image.open("5.jpg")
#preprocess_img(img,"results.json")
#img = img.resize((100, 100))
#img=Image.open("5.jpg")
#print(img.size)
predictor =UVPredictor(img_path,tex_path,tensor)
#predictor.predict(F.interpolate(uv_tensor.unsqueeze(0) ,size=target_size, mode='bilinear', align_corners=False))
# predictor.combine_blocks()
predictor.predict()
