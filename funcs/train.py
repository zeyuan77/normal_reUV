
import torch

def compute_d_x(ori):
    temp=ori[1:,:]-ori[:-1,:]
    d_x=torch.zeros_like(ori)
    d_x[1:,:]=temp
    return d_x

def compute_d_y(ori):
    temp=ori[:,1:]-ori[:,:-1]
    d_y=torch.zeros_like(ori)
    d_y[:,1:]=temp
    return d_y

def uv_normal_loss(uv, normal, pre_uv):
    #uv.shape(1,2,h,w)
    #normal.shape(1,3,h,w) dim:(x,y,z)->(r,g,b)

    du_x=compute_d_x(uv[0,0])
    dv_x=compute_d_x(uv[0,1])
    du_y=compute_d_y(uv[0,0])
    dv_y=compute_d_y(uv[0,1])
    # print(du_x.shape,dv_x.shape,du_y.shape,dv_y.shape)
    
    nx=normal[0,0,:]
    ny=normal[0,1,:]
    nz=normal[0,2,:]

    epsilon = 1e-8
    loss_geo=(du_x**2+dv_x**2-1-nx**2/(nz**2+epsilon))**2+(du_y**2+dv_y**2-1-ny**2/(nz**2+epsilon))**2+(du_x*du_y+dv_x*dv_y-1-nx*ny/(nz**2+epsilon))**2
    loss_prox=(uv-pre_uv)**2
    loss_z=torch.clamp(du_x * dv_y - du_y*dv_x, min=0)

    loss_fin=loss_geo.sum()+0.2*loss_prox.sum()+0.01*loss_z.sum()
    return loss_fin


def train_main(imgDataset, model, updater, num_epochs):
    uv=imgDataset.initUV()
    normal=imgDataset.getImgTensor("normal")
    normal=imgDataset.byMask(normal)
    pre_uv=imgDataset.getPreUV()

    for epoch in range(num_epochs):
        resUV=model(uv)
        resUV=imgDataset.byMask(resUV)

        updater.zero_grad()
        loss=uv_normal_loss(resUV, normal, pre_uv)
        loss.backward()
        updater.step()
        if(epoch%10==0 or epoch==num_epochs-1):
            print("Epoch:", epoch)
            print("Train_loss",loss)
    return resUV
        
