"""
Author: LiGorden
Email: likehao1006@gmail.com
URL: https://www.linkedin.com/in/kehao-li-06a9a2235/
ResearchGate: https://www.researchgate.net/profile/Gorden-Li

Predict the result using ResNet
"""
import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

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

    """ For prediction of 1 image
    img_path = "../../Image/t-SNE/Image_t_SNE_V2.2/val/Positive/Positive_Test_3.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    """
    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = resnet34(num_classes=2).to(device)

    # load model weights, 导入训练好的权重
    weights_path = "./resNet34.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    prediction_positive = list()
    for img_path in img_path_list_positive:
        print(img_path)
        raw_img = Image.open(img_path)
        # plt.imshow(rawimg)
        # [N, C, H, W]
        img = data_transform(raw_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            if predict_cla == 0:
                prediction_positive.append([predict_cla,  predict[predict_cla].numpy(), img_path])
                print_res = "class: {}  prob: {:.3}".format(class_indict[str(predict_cla)], predict[predict_cla].numpy())
                plt.figure()
                plt.title(print_res)
                plt.imshow(raw_img)
                save_path = 'C:/Personal/MachineLearning/Python/Project/Interpretable AI/Data2Image/Image/t-SNE/Fault/False Negative/' + str(predict[predict_cla].numpy()) + '.png'
                plt.savefig(save_path)

    prediction_negative = list()
    for img_path in img_path_list_negative:
        print(img_path)
        raw_img = Image.open(img_path)
        # plt.imshow(raw_img)
        # [N, C, H, W]
        img = data_transform(raw_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            if predict_cla == 1:
                prediction_negative.append([predict_cla,  predict[predict_cla].numpy(), img_path])
                print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)], predict[predict_cla].numpy())
                plt.figure()
                plt.title(print_res)
                plt.imshow(raw_img)
                save_path = 'C:/Personal/MachineLearning/Python/Project/Interpretable AI/Data2Image/Image/t-SNE/Fault/False Positive/' + str(predict[predict_cla].numpy()) + '.png'
                plt.savefig(save_path)

    return prediction_positive

        # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)], predict[predict_cla].numpy())
    # plt.title(print_res)
    # print(print_res)
    # plt.show()


if __name__ == '__main__':
    positive, negative = main()
