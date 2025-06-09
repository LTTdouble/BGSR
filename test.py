import os
import numpy as np
import torch
from os import listdir
from scipy.stats import pearsonr
from torch.autograd import Variable
from data_utils import is_image_file
from model import bgsr
from option import opt
import scipy.io as scio
from eval import PSNR, SSIM, SAM,compare_corr
from thop import profile

def pad_and_group_channels(image):

    group_size =7
    B, N, H, W = image.shape

    remainder = N % group_size

    if remainder != 0:
        pad_size = group_size - remainder

        pad = image[:, -pad_size - 1:-1, :]
        image = torch.cat([image, pad],1)

    else:
        image =image

    return image

def calculate_corr_matrix(input):
    corr_matrix = torch.zeros(input.size(1), input.size(1))
    for i in range(input.size(1)):
        for j in range(input.size(1)):
            corr, _ = pearsonr(input[:, i].numpy().flatten(), input[:, j].numpy().flatten())
            corr = np.clip(corr, -1.0, 1.0)
            corr_matrix[i, j] = corr
    return corr_matrix


def group_channels(input, threshold, corr_matrix):
    groups = []
    visited = set()
    for i in range(input.size(1)):
        if i not in visited:
            group = [i]
            visited.add(i)
            for j in range(i + 1, input.size(1)):
                if j not in visited and corr_matrix[i, j] > threshold:
                    group.append(j)
                    visited.add(j)
            groups.append(group)
    return groups

def main( ):
    input_path = './dataset/tests/' + opt.datasetName + '/' + str(
        opt.upscale_factor) + '/'
    out_path = './Result/' + opt.datasetName + '/' + str(
        opt.upscale_factor) + '/' + opt.method + '/'

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    PSNRs = []
    SSIMs = []
    SAMs = []
    CORs = []

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if opt.cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    model =bgsr.BGSR(nf=opt.n_feats,front_RBs=3)

    if opt.cuda:
        model = model.cuda()

    checkpoint = torch.load(opt.model_name)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    images_name = [x for x in listdir(input_path) if is_image_file(x)]

    T = 0
    with torch.no_grad():
        for index in range(len(images_name)):
            group_size=7

            mat = scio.loadmat(input_path + images_name[index])

            # LR = mat['lr'].astype(np.float32).transpose(2, 0, 1)
            # HR = mat['hr'].astype(np.float32).transpose(2, 0, 1)

            LR = mat['lr'].astype(np.float32)
            HR = mat['hr'].astype(np.float32)

            input = Variable(torch.from_numpy(LR).float(), volatile=True).contiguous().view(1, -1, LR.shape[1], LR.shape[2])

            HR = Variable(torch.from_numpy(HR).float(), volatile=True).contiguous().view(1, -1, HR.shape[1], HR.shape[2])
            input1 = pad_and_group_channels(input)

            num_channels = input1.shape[1]
            split_points = range(0, num_channels, group_size)
            splitted_images = np.array_split(input1, split_points, axis=1)

            if opt.cuda:

                input1 = input1.cuda()
                input = input.cuda()
                HR = HR.cuda()

            SR, f = model(input,input1, splitted_images)

            SR = SR.cpu().data[0].numpy().astype(np.float32)
            HR = HR.cpu().data[0].numpy().astype(np.float32)

            f = f.cpu().data[0].numpy().astype(np.float32)

            psnr = PSNR(SR, HR)
            ssim = SSIM(SR, HR)
            sam = SAM(SR, HR)
            cor = compare_corr(SR, HR)

            PSNRs.append(psnr)
            SSIMs.append(ssim)
            SAMs.append(sam)
            CORs.append(cor)
            SR = SR.transpose(1, 2, 0)
            HR = HR.transpose( 2,1, 0)

            f = f.transpose(2, 1, 0)
            torch.cuda.empty_cache()

            scio.savemat(out_path + images_name[index], {'HR': HR, 'SR': SR,'first_result': f})
            print("===The {}-th picture=====PSNR:{:.3f}=====SSIM:{:.4f}=====SAM:{:.3f}=====COR:{:.3f}====Name:{}".format(index + 1, psnr, ssim, sam,cor,images_name[ index]))
    print("=====averPSNR:{:.3f}=====averSSIM:{:.4f}=====averSAM:{:.3f}======averCOR:{:.3f}".format(np.mean(PSNRs), np.mean(SSIMs),np.mean(SAMs),np.mean(CORs)))



if __name__ == "__main__":
    main()