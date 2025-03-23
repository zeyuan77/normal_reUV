import json
from PIL import Image
with open("E:\\深度学习\\normal_reUV\\readbox\\img1Json.json", 'r', encoding='utf-8') as f_densepose:
    json_str = json.load(f_densepose)  # str
denseposeDict = json.loads(json_str)  # dict
pred_boxes_XYXY=denseposeDict['pred_boxes_XYXY']
#print(type(pred_boxes_XYXY))#list
image=Image.open("E:\深度学习\\normal_reUV\\readbox\\imgtoRead.jpg")
left=pred_boxes_XYXY[0][0]
top=pred_boxes_XYXY[0][1]
right=pred_boxes_XYXY[0][2]
bottom=pred_boxes_XYXY[0][3]
box=(left,top,right,bottom)
cropped_image=image.crop(box)
cropped_image.save("E:\深度学习\\normal_reUV\\readbox\\cropped_image.jpg")