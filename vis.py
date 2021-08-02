from mmdet.apis import init_detector, inference_detector
import mmcv
import os


def main():
    config_file= 'detect_config.py'
    checkpoint_file='/apdcephfs/share_1290939/jiaxiaojun/OpenBrandData/outputEnd_detectors_r50/epoch_21.pth'
    #checkpoint_file='/apdcephfs/share_1290939/jiaxiaojun/ICCV2021Challenges/Detection/output/detectors-trainval-r50-c10-raw/epoch_37.pth'
    #checkpoint_file = '/apdcephfs/share_1290939/jiaxiaojun/ICCV2021Challenges/Detection/output/detectors-trainval-r50-c10-raw/epoch_37.pth'
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    imgdir='/apdcephfs/share_1290939/jiaxiaojun/OpenBrandData/mini-testB-images/'
    cnt=0
    for img in os.listdir(imgdir):
        #print(img)
        img=os.path.join(imgdir,img)
        result = inference_detector(model, img)
        # visualize the results in a new window
        #show_result(img, result, model.CLASSES)
        # or save the visualization results to image files
        model.show_result(img, result, out_file='results/result{}.jpg'.format(cnt))
        cnt+=1
main()