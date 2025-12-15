
This project primarily implements a crowd counting
framework. Learners can use this framework with
a single click to generate density map datasets,
and the framework also provides implementations
of some common loss functions that can be directly 
integrated and used. The most notable feature of
this framework is that learners do not need to 
modify any code to train models. If learners 
wish to incorporate their own models, they can 
simply implement them and embed them into this 
training framework. This framework allows learners 
to focus solely on the core aspects without worrying 
about other peripheral code, thereby helping them
concentrate more on related algorithms. 
Additionally, it supports distributed training 
of models. References to other open-source code
used in this project have been provided with 
links at the end of the project.


Github: [https://github.com/KeepTryingTo](https://github.com/KeepTryingTo)

Bilibili: [https://space.bilibili.com/625095571?spm_id_from=333.1007.0.0](https://space.bilibili.com/625095571?spm_id_from=333.1007.0.0)

CSDN: [https://blog.csdn.net/Keep_Trying_Go?type=blog](https://blog.csdn.net/Keep_Trying_Go?type=blog)

DouYin: [https://www.douyin.com/user/self](https://www.douyin.com/user/self)

### Support
#### Training Configuration

[√] Single GPU, CPU or Multi-GPU Distributed Training

[√] Logging and Model Checkpointing​ (Save best performing model)

[√] GPU Utilization Statistics

#### Data Processing

[√] Custom Bounding Box Dataset to Density Map Generation

[√] One-click Density Map Generation​ for common datasets including:
    **ShangHai_partA, ShangHai_partB,NWPU, QNRF, UCF_CC_50,JHUC-CROWD++**

#### Model Architecture

[√] Pre-built Backbone Network Models

[√] Comprehensive Loss Function Library

#### Model Analysis

[√] Model Parameter Count and FLOPs Statistics

[√] MAE Training Curve Visualization

### Visualization Tools

[√] Ground Truth Density Map Visualization​ (npy format)

[√] Model Predicted Density Map Visualization

#### Optimization

[√] Built-in Learning Rate Schedulers

### Project Structure
```angular2html
.
├── ./CLIP
│   ├── ./CLIP/clip
│   ├── ./CLIP/CLIP.png
│   ├── ./CLIP/data
│   ├── ./CLIP/demo.py
│   ├── ./CLIP/hubconf.py
│   ├── ./CLIP/images
│   ├── ./CLIP/model-card.md
│   ├── ./CLIP/notebooks
│   ├── ./CLIP/README.md
│   ├── ./CLIP/requirements.txt
│   ├── ./CLIP/setup.py
│   └── ./CLIP/tests
├── ./configs
│   ├── ./configs/config_lr.py
│   └── ./configs/config.py
├── ./datasets
│   ├── ./datasets/copy.py
│   ├── ./datasets/crop_images.py
│   ├── ./datasets/dataset
│   ├── ./datasets/dmap_process
│   ├── ./datasets/__init__.py
│   ├── ./datasets/jhu_crowd_v2.0
│   ├── ./datasets/process_data
│   ├── ./datasets/__pycache__
│   ├── ./datasets/rename.py
│   ├── ./datasets/unit_process_data.py
│   └── ./datasets/utils
├── ./losses
│   ├── ./losses/bregman_pytorch.py
│   ├── ./losses/mae_loss.py
│   ├── ./losses/mae.py
│   ├── ./losses/MMD.py
│   ├── ./losses/mse_loss.py
│   ├── ./losses/mse.py
│   ├── ./losses/ot_loss.py
│   ├── ./losses/pytorch_ssim
│   ├── ./losses/ssim_loss.py
│   └── ./losses/transforms.py
├── ./main.py
├── ./main_ucf_cc_50.py
├── ./models
│   ├── ./models/backbones
│   ├── ./models/crowd_convnext.py
│   ├── ./models/crowd_efficientnetvx.py
│   ├── ./models/crowd_mobilenetv1.py
│   ├── ./models/crowd_mobilenetv2.py
│   ├── ./models/crowd_mobilenetv3.py
│   ├── ./models/crowd_resnet_pretrained.py
│   ├── ./models/crowd_resnet.py
│   ├── ./models/crowd_shufflenetv2.py
│   ├── ./models/crowd_vgg.py
│   ├── ./models/heads
│   ├── ./models/load_model.py
│   └── ./models/utils
├── ./plot.py
├── ./README.md
├── ./structure.txt
├── ./test.py
├── ./trainer.py
├── ./utils
│   ├── ./utils/distributed.py
│   ├── ./utils/evaluator.py
│   ├── ./utils/flops.py
│   ├── ./utils/gpu_usage.py
│   ├── ./utils/lr_schedular.py
│   ├── ./utils/save_checkpoint.py
│   ├── ./utils/select_devices.py
│   ├── ./utils/sliding_window.py
│   └── ./utils/util.py
├── ./visDemo.py
├── ./vis_gt.py
└── ./weights
```

### Environment
Actually, there are no strict requirements
for the versions of torch and torchvision 
here, as no highly specialized libraries 
or similar dependencies are used.
```doctest
torch                     1.11.0+cu115
torchmetrics              1.5.2
torchtext                 0.12.0
torchvision               0.12.0+cu115
```


### One-Click Dataset Processing
```angular2html
cd datasets
```
Note: The one-click dataset generation feature 
only requires specifying the root directory for
downloading the dataset and the root directory
for saving the processed dataset.  
Please note that the dataset name must be specified. 
All generated datasets can be saved to the same 
directory. For example, if the specified save 
directory is **./process_data**, the generated 
dataset structure under this root directory will
be as follows:
```doctest
.
├── jhu
│   ├── test
│   │   ├── imgs
│   │   └── npys
│   ├── train
│   │   ├── imgs
│   │   └── npys
│   └── val
│       └── npys
├── nwpu
│   ├── test
│   │   └── imgs
│   ├── train
│   │   ├── imgs
│   │   └── npys
│   └── val
│       ├── imgs
│       └── npys
├── qnrf
│   ├── test
│   │   ├── imgs
│   │   └── npys
│   └── train
│       ├── imgs
│       └── npys
├── sha
│   ├── test
│   │   ├── imgs
│   │   └── npys
│   └── train
│       ├── imgs
│       └── npys
├── shb
│   ├── test
│   │   ├── imgs
│   │   └── npys
│   └── train
│       ├── imgs
│       └── npys
└── ucf_cc_50
    ├── imgs
    └── npys
```
```doctest
# ./datasets/unit_process_data.py
managerDataset = ManagerDataset(
        dataset_name='jhu',
        root_dir=r'数据集路径',
        save_dir=r'处理之后数据集保存路径'
    )

managerDataset()
```

### Custom Bounding Box Dataset Generation for Density Maps
```doctest
# into directory
cd datasets
# Generate density maps for custom bounding box datasets
custom_data_generate_npy.py

# The following directories need to be specified in custom_data_generate_npy.py:
# Provide the root directory for images and the directory for bounding box coordinates
img_root = r'./dataset/data02/test/images'
label_root = r'./dataset/data02/test/labels'

# save images and npys directory
save_dir = r'./datasets/peppers/test'
```
Explanation: After entering **custom_data_generate_npy.py**, you can 
specify the supported data formats for the bounding boxes, such 
as **[x1, y1, x2, y2]** or **[center_x, center_y, width, height]**. At the 
same time, you can also specify whether the values of the bounding 
boxes have undergone normalization. For example, the values below 
have been normalized (relative coordinates). They are restored to 
coordinates relative to the original image size by multiplying by 
the height and width of the original image.
```doctest
"""
'./dataset/data02/test/labels/1000.txt'The content format is as follows: [x1, y1, x2, y2]
0.004263484453315814 0.2570867056798453 0.3127755427706069 0.5225552645596591
0.3522711649197723 0.007105183521103778 0.621207290554639 0.22462686866220802
0.7087478795653791 0.6014062643853904 0.9993402390252977 0.8706049549860585
0.3778102708899457 0.2049072599571562 0.7409225685009058 0.4753022884278988
0.16197438368392533 0.6013358787254051 0.4457309951940185 0.8692702990188341
0.5841927538253753 0.30822610052346383 0.8606768305997671 0.5426243740300136
......
"""
```

Note: All configuration information related to model
training, testing, and visualization can be set in 
**./configs/config.py**. In PyCharm, you can directly
right-click to run **training/testing/visualization**. 
Alternatively, you can enter the terminal to perform 
model **training/testing/visualization** using **python 
main.py / test.py / visDemo.py**.

### Model train
```doctest
# Configure relevant parameters in configs/config.py.
python main.py
```

### Model test
```doctest
# Configure relevant parameters in test.py.
python test.py
```

### Image visualize
![](./outputs/gt_pred.png)
```doctest
# visDemo is primarily used for visualizing prediction results. 
# Relevant parameters can be configured in visDemo.py.
python visDemo.py

# vis_gt is primarily used to visualize the ground truth labels 
# stored in .npyfiles. This function loads the label data 
# from the specified NumPy file and generates corresponding 
# visual representations
vis_Gt(
    npy_path_root, 
    save_dir
)
```

说明： The vis_img_den.py visualization script primarily uses 
matplotlib to display the original image, the ground‑truth 
density map, the predicted density map, and the overlaid result 
of the ground‑truth and predicted density maps. Among them, the 
divided_image_patch function in vis_img_den.py needs careful 
understanding. In order to ensure that the original image after 
tiling is processed by the model in patches and the patch‑wise 
predictions are stitched back to the original size, the entire 
tiling process must preserve the tiling indices, so that the 
subsequent stitching can restore the original image dimensions, 
enabling the final overlay of the original image and the predicted 
density map to be performed correctly.
```doctest

# vis_img_den.py

img_root = r'/home/ff/myProject/KGT/myProjects/myDataset/shanghai/ShanghaiTech/part_A/DCCUS_final/test_data/imgs'
npy_root = r'/home/ff/myProject/KGT/myProjects/myDataset/shanghai/ShanghaiTech/part_A/DCCUS_final/test_data/npys'
save_dir = r'/home/ff/myProject/KGT/myProjects/myProjects/CrowdCLIP/myGLPro/datasets/shb/vis'

from datasets.utils import transforms as T
args = get_parser()
device = 'cpu' if torch.cuda.is_available() else 'cpu'

model, optim, start_epoch, best_mae, best_mse = create_model(args, device=device)
model.eval()

normalizer = T.standard_transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
img_transformer = T.standard_transforms.Compose([
    T.standard_transforms.ToTensor(),
    normalizer
])
    
main(img_root, npy_root, save_dir)
```
![](./outputs/vis_img_den.png)


### 绘制MAE和MSE曲线
Note: The plot.py script is primarily used to plot the **MAE 
(Mean Absolute Error)** metric saved during the model training 
process. The parameters of this function are the path list 
of the **mae_mse.txt** files and the label list, where the label 
list specifies the legend names for the plotted **MAE** curves. 
For example, the plotting result is as follows:
![](./outputs/result.png)
```doctest
# plot.py函数
plot_mae(
    paths=[r'./weights/logs/sha/s_512_t_2048_2025-11-21_21-21-46/mae_mse.txt'],
    labels=['MAE']
)
```

### 参考链接
[https://github.com/cvlab-stonybrook/DM-Count]()

[https://github.com/gjy3035/Awesome-Crowd-Counting]()

[https://github.com/gjy3035/C-3-Framework]()

[https://github.com/ZPDu/Domain-general-Crowd-Counting-in-Unseen-Scenarios]()

[https://github.com/openai/CLIP]()

[https://mydreamambitious.blog.csdn.net/article/details/142730047?spm=1011.2415.3001.5331]()

[https://mydreamambitious.blog.csdn.net/article/details/147144537?spm=1011.2415.3001.5331]()

[https://mydreamambitious.blog.csdn.net/article/details/143133438?spm=1011.2415.3001.5331]()

[https://mydreamambitious.blog.csdn.net/article/details/143219789?spm=1011.2415.3001.5331]()

[https://mydreamambitious.blog.csdn.net/article/details/144692297?spm=1011.2415.3001.5331]()

[https://github.com/erdongsanshi/FFNet]()

[https://mydreamambitious.blog.csdn.net/article/details/141355068?spm=1011.2415.3001.5331]()

[https://github.com/KeepTryingTo/PyTorch-DeepLearning-Visual-LLM]()

[https://github.com/KeepTryingTo]()

[https://blog.csdn.net/Keep_Trying_Go/article/details/154913403]()

[https://mydreamambitious.blog.csdn.net/article/details/154789450?spm=1011.2415.3001.5331]()

[https://mydreamambitious.blog.csdn.net/article/details/154443638?spm=1011.2415.3001.5331]()

[https://mydreamambitious.blog.csdn.net/article/details/148124251?spm=1011.2415.3001.5331]()

[https://mydreamambitious.blog.csdn.net/article/details/148300284?spm=1011.2415.3001.5331]()

[https://mydreamambitious.blog.csdn.net/article/details/148227727?spm=1011.2415.3001.5331]()

[https://mydreamambitious.blog.csdn.net/article/details/147873916?spm=1011.2415.3001.5331]()

[https://mydreamambitious.blog.csdn.net/article/details/147851355?spm=1011.2415.3001.5331]()

[https://mydreamambitious.blog.csdn.net/article/details/147819012?spm=1011.2415.3001.5331]()

[]()

[]()

[]()






