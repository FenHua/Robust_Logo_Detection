import os
import torch


def SWA(model_dir, model_names=[], save_dir=None):
    model_dirs = [os.path.join(model_dir, mname) for mname in model_names]  # 每个模型的的路径
    models = [torch.load(model_dir) for model_dir in model_dirs]  # 加载网络模型
    model_num = len(models)    # 模型数
    model_keys = models[-1]['state_dict'].keys()  # 模型关键字
    state_dict = models[-1]['state_dict']         #
    new_state_dict = state_dict.copy()
    ref_model = models[-1]                        # 新的网络模型初始化

    # swa
    for key in model_keys:
        sum_weight = 0.0
        for m in models:
            sum_weight += m['state_dict'][key]
        avg_weight = sum_weight / model_num
        new_state_dict[key] = avg_weight
    ref_model['state_dict'] = new_state_dict  # 随机权重平均后的权重
    save_model_name = 'swa_weights' + '.pth'
    if save_dir is not None:
        save_dir = os.path.join(save_dir, save_model_name)
    else:
        save_dir = os.path.join(model_dir, save_model_name)
    torch.save(ref_model, save_dir)          # 保存网络权重
    print('Model is saved at', save_dir)


if __name__ == '__main__':
    model_dir = 'weights'
    model_names = ['epoch_21.pth','epoch_22.pth','epoch_23.pth','epoch_24.pth','epoch_25.pth',
                   'epoch_26.pth','epoch_27.pth']
    SWA(model_dir,model_names)