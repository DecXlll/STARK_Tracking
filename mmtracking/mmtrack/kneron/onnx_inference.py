import os, sys
sys.path.append(os.getcwd())
import onnxruntime
import onnx
import cv2
import torch
import torchvision.models as models
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
class onnx_inference():
    def to_numpy(self,tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def get_test_transform(self):
        return transforms.Compose([
            transforms.Resize([320, 320]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
            ])
    def inference(self,img=None,i=1):
        if (img==None):
            image = Image.open('t.jpg') # 289
            img = self.get_test_transform()(image)/255.
            img = img.unsqueeze_(0) # -> NCHW, 1,3,224,224
        else:
            img_ = tuple(t.cpu().numpy() for t in img)
            img_ = np.asarray(img_).copy()
            # img_.transpose((0,1,3,2))
        print(np.array(img_).shape)
        print("input img mean {} and std {}".format(img.mean(), img.std()))

        # onnx_model_path = "base_backbone320x320.onnx"
        onnx_model_path = "base_backbone320x320_optimized.onnx"
        # pth_model_path = "resnet18.pth"

        ## Host GPU pth测试
        # resnet18 = models.resnet18()
        # net = resnet18
        # net.load_state_dict(torch.load(pth_model_path))
        # net.eval()
        # output = net(img)

        # print("pth weights", output.detach().cpu().numpy())
        # print("HOST GPU prediction", output.argmax(dim=1)[0].item())

        ##onnx测试
        resnet_session = onnxruntime.InferenceSession(onnx_model_path)
        #compute ONNX Runtime output prediction
        inputs = {resnet_session.get_inputs()[0].name:self.to_numpy(img)}
        outs = resnet_session.run(None, inputs)[0]
        print("onnx shape:",np.array(outs).shape)
        outs = np.array(outs).reshape((1,1,1024,20,20))
        # print("onnx weights", outs,np.array(outs).shape)
        # print("onnx prediction", outs.argmax(axis=1)[0])

        # path = f'mmtrack/onnx_backbone{i}.txt'
        # f = open(path, 'w')
        # f.write("shape")
        # f.write(str(np.array(outs).shape))
        # f.write(str(outs))
        # f.close()
        tensor_result= torch.tensor(outs,dtype=torch.float, device=torch.device('cuda:0'))
        return tensor_result
        

# o = onnx_inference()
# o.inference()