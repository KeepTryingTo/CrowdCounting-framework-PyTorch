"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/4/24-15:17
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import clip
import torch
import numpy as np
from PIL import Image

import os
import skimage
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def predict_one():
    model,preprocess = clip.load(name='ViT-B/32',device=device)
    image= preprocess(
        Image.open('images/dog01.png')
    ).unsqueeze(dim=0).to(device)
    text = clip.tokenize(['a dog','two dogs','a cat','a horse']).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image,logits_per_text = model(image,text)
        image_probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        text_probs = logits_per_text.softmax(dim=-1).cpu().numpy()
    print('most likely label probs: {}'.format(image_probs))
    print('most likely text probs: {}'.format(text_probs))

def predict_more():
    clip.available_models()
    model, preprocess = clip.load("ViT-B/32")
    model.cuda().eval()
    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    print('process image: {}'.format(preprocess))

    print(clip.tokenize(texts='i love my life and only love once'))


    # images in skimage to use and their textual descriptions
    descriptions = {
        "bicycle01": "In the front of building,there is a man and sit down the bycycle",
        "bicycle02": "On the road,there are many cars and persons that they bycycle riding cross the road,and ",
        "bicycle03": "In the sea side,there are two man bycycle riding",
        "bicycle04": "In the sea side and on the road,there are a man and a child bycycle,they look at very happy",
        "bicycle05": "There is a girl bycycle riding and performance skills",
        "bicycle06": "A old person bycycle riding quickly",
        "bottle01": "There are four bottles and inside a bottle with flower",
        "bottle02": "There are three transparent bottles",
        "bottle03":"There are fours different colors bottles",
        "dog01":"Two dogs in the flowers",
        "dog02":"In the river side,a dog with smile",
        "horse01":"On the grassland,a horse is browsing",
        "person01":"A woman is appreciating natual",
        "person02":"A woman is standing in the flowers",
        "yolov1":"A dog and bycycle,in the distance,there is a car"
    }
    original_images = []
    images = []
    texts = []
    plt.figure(figsize=(16, 5))

    for filename in [filename for filename in os.listdir(r'images') if
                     filename.endswith(".png") or filename.endswith(".jpg")]:
        name = os.path.splitext(filename)[0]
        #判断当前的图像是否存在对应的内容描述
        if name not in descriptions:
            continue

        image = Image.open(os.path.join(r'images', filename)).convert("RGB")

        plt.subplot(4, 4, len(images) + 1)
        plt.imshow(image)
        plt.title(f"{filename}\n{descriptions[name]}")
        plt.xticks([])
        plt.yticks([])

        original_images.append(image)
        images.append(preprocess(image))
        texts.append(descriptions[name])

    plt.tight_layout()
    plt.savefig(f'runs/layout.png')
    print('------------------------------------------------------------------------')

    image_input = torch.tensor(np.stack(images)).to(device)
    text_tokens = clip.tokenize(["This is " + desc for desc in texts]).to(device)

    #计算相似性
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_tokens).float()

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

    count = len(descriptions)

    print('------------------------------------------------------------------------')
    plt.figure(figsize=(20, 14))
    plt.imshow(similarity, vmin=0.1, vmax=0.3)

    plt.yticks(range(count), texts, fontsize=18)
    plt.xticks([])
    for i, image in enumerate(original_images):
        plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)
    """
    plt.gca：获取坐标轴信息
    ax.spines：选择边框
    set_color：设置颜色
    """
    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)
    """
    plt.xlim() 显示的是x轴的作图范围
    plt.ylim() 显示的是y轴的作图范围
    """
    plt.xlim([-0.5, count - 0.5])
    plt.ylim([count + 0.5, -2])
    plt.title("Cosine similarity between text and image features", size=20)
    plt.savefig('runs/similarity.png')
    print('------------------------------------------------------------------------')

    from torchvision.datasets import CIFAR100
    cifar100 = CIFAR100(root=r'dataset/',transform=preprocess, download=True)
    #得到100个类别名称并使用文本叙述表示
    text_descriptions = [f"This is a photo of a {label}" for label in cifar100.classes]
    text_tokens = clip.tokenize(text_descriptions).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    #计算输入的图像和CIFAR100中对应图像的文本相似性，从而判断出CIFAR100中的图像和给定的图像之间的相似性
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    #得到和图像概率最大的前5个作为最终结果
    top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)

    plt.figure(figsize=(16, 16))
    for i, image in enumerate(original_images):
        plt.subplot(6,6, 2 * i + 1)
        plt.imshow(image)
        plt.axis("off")

        plt.subplot(6, 6, 2 * i + 2)
        y = np.arange(top_probs.shape[-1])
        plt.grid()
        #plt.barh()：横向的柱状图，可以理解为正常柱状图旋转了90°。
        plt.barh(y, top_probs[i]) #绘制前5个分数最高的条形统计图
        """
        plt.gca().invert_yaxis() 是一个用于反转y轴坐标的函数。
            它是matplotlib库中的一个函数，用于将y轴坐标从上到下反转，
            即将最大值放在底部，最小值放在顶部。
        plt.gca().set_axisbelow(True) 是一个用于将坐标轴置于
            图形下方的函数。它也是matplotlib库中的一个函数，用于将坐标
            轴放置在图形的下方，使得图形中的元素可以覆盖在坐标轴之上。
        """
        plt.gca().invert_yaxis()
        plt.gca().set_axisbelow(True)
        #获得前5个概率最大的对应类别标签
        plt.yticks(y, [cifar100.classes[index] for index in top_labels[i].numpy()])
        plt.xlabel("probability")

    plt.subplots_adjust(wspace=0.5)
    plt.savefig(r'runs/cifar100.png')
    # plt.show()

if __name__ == '__main__':
    predict_more()







