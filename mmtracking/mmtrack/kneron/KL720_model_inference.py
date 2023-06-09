from array import array
import imp
import os
import sys
import argparse
import numpy as np
# np.set_printoptions(threshold=np.inf)
import torch
import math
from torchvision.transforms.functional import normalize

PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(PWD, '..'))

from .utils_PLUS_1_3_0.ExampleHelper import get_device_usb_speed_by_port_id

import kp
import cv2

class kneron_run():
    def __init__(self,device_group=None):
        self.device_group = device_group

        usb_port_id = 0
        """
        check device USB speed (Recommend run KL720 at super speed)
        """
        try:
            if kp.UsbSpeed.KP_USB_SPEED_SUPER != get_device_usb_speed_by_port_id(usb_port_id=usb_port_id):
                print('\033[91m' + '[Error] Device is not run at super speed.' + '\033[0m')
                exit(0)
        except Exception as exception:
            print('Error: check device USB speed fail, port ID = \'{}\', error msg: [{}]'.format(usb_port_id,
                                                                                                str(exception)))
            exit(0)

        """
        connect the device
        """
        try:
            print('[Connect Device]')
            self.device_group = kp.core.connect_devices(usb_port_ids=[usb_port_id])
            print(' - Success')
        except kp.ApiKPException as exception:
            print('Error: connect device fail, port ID = \'{}\', error msg: [{}]'.format(usb_port_id,
                                                                                        str(exception)))
            exit(0)

        """
        setting timeout of the usb communication with the device
        """
        print('[Set Device Timeout]')
        kp.core.set_timeout(device_group=self.device_group, milliseconds=10000)
        print(' - Success')


        
    def upload_model(self, model_path=None, from_flash=False):

        if(from_flash):
            try:
                print('[Load Model from Flash]')
                self.model_nef_descriptor = kp.core.load_model_from_flash(device_group=self.device_group)
                print(' - Success')
            except kp.ApiKPException as exception:
                print('Error: load model from device flash failed, error = \'{}\''.format(str(exception)))
                exit(0)

        else:
            MODEL_FILE_PATH = os.path.join(PWD,model_path)
            print("MODEL_FILE_PATH:",MODEL_FILE_PATH)
            try:
                print('[Upload Model]')
                self.model_nef_descriptor = kp.core.load_model_from_file(device_group=self.device_group,
                                                                    file_path=MODEL_FILE_PATH)
                print(' - Success')
            except kp.ApiKPException as exception:
                print('Error: upload model failed, error = \'{}\''.format(str(exception)))
                exit(0)   

        
        

           
       
        
    def inference_model(self,img,BypassPreProc=True):
        img_ = tuple(t.cpu().numpy() for t in img)
        # feat_backbone = np.asarray(feat_backbone).copy()
        img_ = np.asarray(img_,np.float16).copy()
        # print("img_:",img_)
        # print(img_.shape)
        img_re = np.transpose(img_,(0,2,3,1)).squeeze() #(1,z,x,y)->(1,x,y,z)
        # print(img_)
        # print(img_re.shape)
        # print(img_re)
        # img_ = cv2.cvtColor(src=img_, code=cv2.COLOR_BGR2BGR565)

        """
        prepare app generic inference config
        """
        
        if(BypassPreProc):
            self.generic_raw_image_header = kp.GenericRawBypassPreProcImageHeader(
                model_id=self.model_nef_descriptor.models[0].id,
                image_buffer_size=len(img_re),
                inference_number=0
            )
            try:
                kp.inference.generic_raw_inference_bypass_pre_proc_send(device_group = self.device_group,
                                            generic_raw_image_header = self.generic_raw_image_header,
                                            image_buffer=img_re)
                generic_result = kp.inference.generic_raw_inference_bypass_pre_proc_receive(device_group = self.device_group,
                                                                                generic_raw_image_header = self.generic_raw_image_header,
                                                                                model_nef_descriptor = self.model_nef_descriptor)
            except kp.ApiKPException as exception:
                print(' - Error: inference failed, error = {}'.format(exception))
                exit(0)

        else:
            self.generic_raw_image_header = kp.GenericRawImageHeader(
                model_id=self.model_nef_descriptor.models[0].id,
                image_format=kp.ImageFormat.KP_IMAGE_FORMAT_RAW8,
                resize_mode=kp.ResizeMode.KP_RESIZE_ENABLE,
                padding_mode=kp.PaddingMode.KP_PADDING_CORNER,
                normalize_mode=kp.NormalizeMode.KP_NORMALIZE_DISABLE,
                inference_number=0
            )
    
            try:
                kp.inference.generic_raw_inference_send(device_group = self.device_group,
                                            generic_raw_image_header = self.generic_raw_image_header,
                                            image=img_re,
                                            image_format=kp.ImageFormat.KP_IMAGE_FORMAT_RAW8)

                generic_result = kp.inference.generic_raw_inference_receive(device_group = self.device_group,
                                                                                generic_raw_image_header = self.generic_raw_image_header,
                                                                                model_nef_descriptor = self.model_nef_descriptor)
            except kp.ApiKPException as exception:
                print(' - Error: inference failed, error = {}'.format(exception))
                exit(0)



        inf_node_output_list = []
        for node_idx in range(generic_result.header.num_output_node):
            inference_float_node_output = kp.inference.generic_inference_retrieve_float_node(node_idx=node_idx,
                                                                                         generic_raw_result=generic_result,
                                                                                         channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_CHW)
        inf_node_output_list.append(inference_float_node_output.ndarray)
        result = np.asarray(inf_node_output_list)
        # print("kneron result:",result.shape)

        tensor_result= torch.tensor(result,dtype=torch.float, device=torch.device('cuda:0'))
        # tensor_result= torch.tensor(result,dtype=torch.float, device=torch.device('cpu'))
        return tensor_result