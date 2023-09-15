from efficientvit.seg_model_zoo import create_seg_model
import time
import numpy as np
import torchvision.transforms as T
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models
# ref: https://learnopencv.com/pytorch-for-beginners-semantic-segmentation-using-torchvision/

imgname = 'dataset/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png'


def decode_segmap(image, nc=21):

    label_colors = np.array([(0, 0, 0),  # 0=background
                             # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                             (128, 0, 0), (0, 128, 0), (128, 128,
                                                        0), (0, 0, 128), (128, 0, 128),
                             # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                             (0, 128, 128), (128, 128, 128), (64,
                                                              0, 0), (192, 0, 0), (64, 128, 0),
                             # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                             (192, 128, 0), (64, 0, 128), (192, 0,
                                                           128), (64, 128, 128), (192, 128, 128),
                             # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                             (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


def segment(net, path, show_orig=True, dev='cuda', size=640):
    img = Image.open(path)
    if show_orig:
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    # Comment the Resize and CenterCrop for better inference results
    trf = T.Compose([T.Resize(size),
                     # T.CenterCrop(224),
                     T.ToTensor(),
                     T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    inp = trf(img).unsqueeze(0).to(dev)
    out = net.to(dev)(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    rgb = decode_segmap(om)
    plt.imshow(rgb)
    plt.axis('off')
    plt.show()


def infer_time(net, path=imgname, dev='cuda', size=224):
    img = Image.open(path)
    trf = T.Compose([T.Resize(size),
                    #  T.CenterCrop(224),
                     T.ToTensor(),
                     T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    inp = trf(img).unsqueeze(0).to(dev)

    st = time.time()
    out1 = net.to(dev)(inp)
    et = time.time()

    return et - st


def calc_ave_time(net, avg_over=10):
    infer_time_list = [infer_time(net) for _ in range(avg_over)]
    infer_time_avg = sum(infer_time_list) / avg_over
    print('The Average Inference time on {} is: {:.3f}s'.format(
        net.__class__.__name__, infer_time_avg))


# load models
fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()
dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()
effvit = create_seg_model(
    name="b0",
    dataset="cityscapes",
    pretrained=True,
    weight_url="assets/checkpoints/seg/cityscapes/b0.pt"
).eval()

# run inference
segment(fcn, imgname)
segment(dlab, imgname)
# segment(effvit, imgname)

# calculate computation time On CUDA
calc_ave_time(fcn)
calc_ave_time(dlab)
