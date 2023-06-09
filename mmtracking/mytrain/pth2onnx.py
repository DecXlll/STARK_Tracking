import torch
from torchvision import datasets, transforms, models 
import cv2
from PIL import Image
import numpy as np

# ##===============================================================================
# ##open resnet50 and Remove dense, flat layers then save
# ##===============================================================================
# stark = torch.load("checkpoints\stark_st2_r50_50e_lasot.pth")
# # print(stark['state_dict'])
# print(stark['meta'].keys())
# print(stark['meta']['iter'])
# # newmodel = torch.nn.ModuleList(*(list(stark['state_dict'])[:-1])) ##Remove dense, flat layers
# # print(newmodel.get_submodule)
# # torch.save(newmodel,"stark_st2_r50_50e_lasot_no_backbone.pth")

##===============================================================================
##open resnet50 and Remove dense, flat layers then save
##===============================================================================
# model = models.resnet50(pretrained=True)
# # model = torch.load("resnet50-0676ba61.pth")
# newmodel = torch.nn.Sequential(*(list(model.children())[:-3])) ##Remove dense, flat layers
# print(newmodel)
# torch.save(newmodel,"resnet40_no_POOL_FC.pth")


##===============================================================================
##inference model
##===============================================================================
# img = Image.open('t.jpg')# Load the image
# data_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
# image = data_transform(img).unsqueeze(0).cuda()#apply the transformation, expand the batch dimension, and send the image to the GPU

# model = torch.load("resnet50_no_POOL_FC.pth")
# model.cuda()
# model.eval()

# # print(model)
# out = model(image)
# print(out.data.shape)

##===============================================================================
##print vector 
##===============================================================================

# out = out.data.cpu().numpy()
# import matplotlib.pyplot as plt
# fig ,axs= plt.subplots(10,10)
# for i in range(1,11):
#     for j in range(1,11):
#         axs[i-1,j-1].imshow(out[0][(i-1)*10+j])       
# plt.show()

#===============================================================================
#turn model to onnx
#===============================================================================
# import torch.onnx
# pth_model = torch.load("save_model/stark/best.pth", map_location='cpu')
# print(pth_model)
# dummy_input = torch.randn(1, 3 , 128, 128 )


# # print(pth_model.keys())
# torch.onnx.export(pth_model, dummy_input, 'head.onnx', opset_version=11)

# # torch.onnx.export(pth_model, {'feat':0,'mask':0}, 'head.onnx', opset_version=11)


##===============================================================================
##turn model to onnx (dynamic input)
##===============================================================================
# import torch.onnx
# # dynamic_axes = {
# #     'feat': { 1: 'x', 2: 'y',3: 'channel', 4: 'batch'},  # 这么写表示NCHW都会变化
# # }


# pth_model = torch.load("head.pth", map_location='cpu')

# # torch.save(newmodel,"CornerPredictorHead.pth")
# # print(pth_model['StarkHead'])
# dummy_image = [{
#         'feat' : tuple(torch.empty(size=(1,1,256,8,8),dtype=torch.float32).random_(256)), 
#         'mask' : torch.empty(size=(1,128,128),dtype=bool).random_(2)
#         },{
#         'feat' : tuple(torch.empty(size=(1,1,256,8,8),dtype=torch.float32).random_(256)), 
#         'mask' : torch.empty(size=(1,128,128),dtype=bool).random_(2)
#         },
#         # {
#         # 'feat' : tuple(torch.empty(size=(1,1,256,20,20),dtype=torch.float32).random_(256)), 
#         # 'mask' : torch.empty(size=(1,320,320),dtype=bool).random_(2)
#         # }
#         ]
    

# # dummy_input = [x]
# # print(np.array(dummy_input[0]['feat']).shape)

# # for i in dummy_input:
# #     print(i['feat'].shape,i['mask'].shape)
# # dynamic_axes= {'query_embedding' : {0: 'batch'}}
        

# # dummy_input = torch.ones([-1], dtype=torch.int, device='cpu')
# torch.onnx.export(pth_model,  # model being run
#                   dummy_image,  # model input (or a tuple for multiple inputs)
#                   'head.onnx',
#                   export_params=True,
#                   opset_version=11,  # the ONNX version to export the model to
#                   operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
#                 #   do_constant_folding=True,
#                   input_names=['query_embedding'],  # the model's input names
#                   dynamic_axes = dynamic_axes
#                   )

import io
import argparse
import onnx
import onnxruntime
import torch
# from hubconf import detr_resnet50
 
 
class ONNXExporter():
    def List2Array(self,list):
        feat_list = []
        mask_list = []
        for input_gpu in list:
            input = tuple(t.cpu().numpy() for t in input_gpu['feat'][0])
            # feat = [[np.array(input['feat'][0],dtype=np.float32).copy()]]
            feat = [[np.array(input,dtype=np.float32).copy()]]
            feat_list = feat_list + feat
            input = tuple(t.cpu().numpy() for t in input_gpu['mask'])
            mask = [np.array(input,dtype=bool).copy()]
            mask_list = mask_list + mask

        all_list = [{'feat':feat_list, 'mask':mask_list,}]
        return all_list

    def run_model(
        self, 
        pth_path, 
        save_path, 
        inputs_list,
        dynamic_axes=None, 
        do_constant_folding=False,
        output_names=None, input_names=None):
        model = torch.load(pth_path, map_location='cpu')
        model.eval()

        input = self.List2Array(inputs_list)
        torch.onnx.export(model, input, save_path,
                          input_names=input_names, 
                          dynamic_axes=dynamic_axes,
                          output_names=output_names, 
                          export_params=True, 
                          training=False,
                          opset_version=11,
                        #   operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                          do_constant_folding=do_constant_folding
                          )
 
        print(f"[INFO] ONNX model export success! save path: {save_path}")

 
# if __name__ == '__main__': 
#     dummy_input = { 'feat' :[
#         tuple(torch.empty(size=(1,1,256,8,8),dtype=torch.float32).random_(256))]+[
#         # tuple(torch.empty(size=(1,1,256,8,8),dtype=torch.float32).random_(256))]+[
#         tuple(torch.empty(size=(1,1,256,20,20),dtype=torch.float32).random_(256))],
#         'mask':[
#         torch.empty(size=(1,128,128),dtype=bool).random_(2)]+[
#         # torch.empty(size=(1,128,128),dtype=bool).random_(2)]+[
#         torch.empty(size=(1,320,320),dtype=bool).random_(2)
#         ]}
        
 
#     # to onnx
#     onnx_export = ONNXExporter()
#     onnx_export.run_model(
#         pth_path="head_ST1.pth", 
#         save_path="head_ST1.onnx",
#         inputs_list=dummy_input, 
#         input_names=['feat0',
#             'feat1',
#             # 'feat2',
#             'mask0',
#             'mask1',
#             # 'mask2'
#             ], 
#         output_names=[
#             # "pred_logits",
#             "pred_boxes"
#             ], 
#         do_constant_folding=True,
#     )
 