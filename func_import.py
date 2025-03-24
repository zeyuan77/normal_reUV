from pathlib import Path
import sys

partL=['funcs']

def addPart(partList)->None:
    tempDir=Path.cwd()
    rootPath="normal_reUV"
    while 1:
        if tempDir.name==rootPath:
            break
        if tempDir==tempDir.parent:
            raise Exception("rootPath: Lose!")
        tempDir=tempDir.parent
    # print("root_path:",tempDir)
    [sys.path.append(str(tempDir/part)) for part in partList]

addPart(partL)

import dataPr
import train


if __name__=="__main__":
    # addPart(partL)
    print(sys.path)



