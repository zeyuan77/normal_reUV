import argparse
import subprocess
import pickle
import json
from PIL import Image

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str)
    parser.add_argument("image", type=str)
    # print(parser.parse_args())
    return parser.parse_args()

def run_DensePose(imgName, mode):
    img=f"myFile/{imgName}.jpg"
    print("ImgName: ", img)
    ori=Image.open(img)
    print("ImgSize:",ori.size)

    dumpSave=None
    # myCommand=f"python apply_net.py dump configs/densepose_rcnn_R_50_FPN_s1x.yaml myFile/model_final_162be9.pkl myFile/{imgName}.jpg --output myFile/{imgName}dump.pkl -v"

    configName="densepose_rcnn_R_101_FPN_DL_WC2M_s1x.yaml"
    config=f"configs/{configName}"
    
    modelName="model_final_de6e7a.pkl"
    model=f"myFile/{modelName}"

    myCommand = [
        "python", "apply_net.py", mode,
        config,
        model,
        img,
    ]
    
    if(mode=="dump"):
        dumpSave=f"myFile/{imgName}Dump.pkl"
        dumpRun = [
        "--output",
        dumpSave,
        "-v"
        ]
        myCommand+=dumpRun

    elif(mode=="show"):
        showSave=f"myFile/{imgName}Show.png"
        showRun = [
        "bbox,dp_segm",
        "--output",
        showSave,
        "-v"
        ]
        myCommand+=showRun

    else:
        raise Exception("Mode Wrong!")
        
    subprocess.run(myCommand, check=True)#Run
    return dumpSave

def trans_DensePoseChartResultWithConfidences(ori):
    res=[]

    for i, data in enumerate(ori):#dataType:<class 'densepose.structures.chart_result.DensePoseChartResultWithConfidences'>
        tempDict={}
        #print(data.labels.shape, data.uv.shape)#example: torch.Size([1006, 567]) torch.Size([2, 1006, 567])
        print("UV_shape:",data.uv.shape)
        tempDict["labels"]=data.labels.to("cpu").numpy().tolist()
        tempDict["uv"]=data.uv.to("cpu").numpy().tolist()
        # print(len(tempDict["labels"]),len(tempDict["uv"]))
        res.append(tempDict)     
    # print(res)
    return res

def transToJson(ori:dict):
    res={}
    for key,value in ori.items():
        if(key=="scores" or key=="pred_boxes_XYXY"):
            res[key]=value.numpy().tolist()
        elif(key=="pred_densepose"):#type=list, len=2, 
            valueTrans=trans_DensePoseChartResultWithConfidences(value)
            res[key]=valueTrans
        else:
            res[key]=value
    # print(res)
    resJson=json.dumps(res)
    
    return resJson
        
    
def printDumpRes(mode, imgName, dumpSave):
    if(mode!="dump"):
        return
        
    with open(dumpSave, 'rb') as file_dp:
        dpData = pickle.load(file_dp)#type=list, len=1
    dictInData=dpData[0]#type=dict,len=4
    
    resJson=transToJson(dictInData)

    jsonSave=f"myFile/{imgName}Json.json"

    with open(jsonSave,'w') as f_json:
        json.dump(resJson, f_json)



        
def main():
    args=get_args()
    dumpSave=run_DensePose(args.image, args.mode)
    printDumpRes(args.mode, args.image, dumpSave)
    
if __name__=="__main__":
    main()


