"""
Author: LiGorden
Email: likehao1006@gmail.com
URL: https://www.linkedin.com/in/kehao-li-06a9a2235/
ResearchGate: https://www.researchgate.net/profile/Gorden-Li

Predict the result using ResNet batch by batch
"""
import os
import json

import torch
from PIL import Image
from torchvision import transforms

from model import resnet34


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [  # transforms.Resize(256),
         # transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path_positive = os.path.join(data_root, "Image/t-SNE/Image_t_SNE_V2.2/val/Positive")  # flower data set path
    image_path_negative = os.path.join(data_root, "Image/t-SNE/Image_t_SNE_V2.2/val/Negative")
    assert os.path.exists(image_path_positive), "{} path does not exist.".format(image_path_positive)
    assert os.path.exists(image_path_negative), "{} path does not exist.".format(image_path_negative)
    img_path_list_positive = list()
    for img_name in os.listdir(image_path_positive):
        img_path = os.path.join(image_path_positive, img_name)
        img_path_list_positive.append(img_path)

    img_path_list_negative = list()
    for img_name in os.listdir(image_path_negative):
        img_path = os.path.join(image_path_negative, img_name)
        img_path_list_negative.append(img_path)

    # Prepare Image for prediction
    img_list = list()
    for img_path in img_path_list_positive:
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        img = data_transform(img)
        img_list.append(img)
    print('Number of Positive Image: %.0f' % len(img_list))

    for img_path in img_path_list_negative:
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        img = data_transform(img)
        img_list.append(img)
    print('Number of Total Image: %.0f' % len(img_list))

    img_path_list = img_path_list_positive + img_path_list_negative

    # batch img, 使用torch.stack将img打包放到batch中, 然后在之后直接将batch放入到模型中进行预测
    batch_img = torch.stack(img_list, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = resnet34(num_classes=2).to(device)

    # load model weights
    weights_path = "./resNet34.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    with torch.no_grad():
        # predict class, 这里直接以batch为单位放入img
        output = model(batch_img.to(device)).cpu()
        predict = torch.softmax(output, dim=1)
        probs, classes = torch.max(predict, dim=1)

        for idx, (pro, cla) in enumerate(zip(probs, classes)):
            print("image: {}  class: {}  prob: {:.3}".format(img_path_list[idx],
                                                             class_indict[str(cla.numpy())],
                                                             pro.numpy()))


if __name__ == '__main__':
    main()
