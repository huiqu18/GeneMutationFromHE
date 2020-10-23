import os
import openslide
from openslide import open_slide
from skimage import io
import numpy as np
from tqdm import tqdm
import pandas as pd
import time
import argparse


''' global variable '''
dataset = 'TCGA-BRCA'
svs_data_dir = '../data/{:s}/slides'.format(dataset)
img_data_dir = '../data/{:s}/20x_512x512'.format(dataset)
thresh_high = 220.0/255  # threshod to remove blank patches
patch_size = 512    # the size of patches extracted from 20x svs image
target_mag = 20
resize_flag = False    # True: downsample the patch from 40x image by 2

# mag_missing_dict = {'TCGA-OL-A5RU-01Z-00-DX1': 20, 'TCGA-OL-A5RV-01Z-00-DX1': 20, 'TCGA-OL-A5RW-01Z-00-DX1': 20,
#                     'TCGA-OL-A5RX-01Z-00-DX1': 20, 'TCGA-OL-A5RY-01Z-00-DX1': 20, 'TCGA-OL-A5RZ-01Z-00-DX1': 20,
#                     'TCGA-OL-A5S0-01Z-00-DX1': 20}

slides_list = pd.read_csv('../data/{:s}/slide_selection_final.txt'.format(dataset), header=None)
slides_list = list(slides_list[0].values)


def main():
    parser = argparse.ArgumentParser(description='The start and end positions in the file list')
    parser.add_argument('--start', type=float, default=0.0, help='start position')
    parser.add_argument('--end', type=float, default=1.0, help='end position')
    args = parser.parse_args()

    if not os.path.exists(img_data_dir):
        os.mkdir(img_data_dir)

    # load slide filenames
    folders = sorted(os.listdir(svs_data_dir))
    N = len(folders)
    folders_to_be_processed = folders[int(N*args.start):int(N*args.end)]

    N_slide = 0
    for folder in folders_to_be_processed:
        files = os.listdir('{:s}/{:s}'.format(svs_data_dir, folder))
        for slide_filename in files:
            if slide_filename[-3:] != 'svs':
                continue
            slide_name = slide_filename.split('.')[0]
            if slide_name not in slides_list:
                continue

            # if slide_name != 'TCGA-DD-AACS-01Z-00-DX1':
            #     continue

            # create folder for each slide sample if it doesn't exist
            sample_folder = '{:s}/{:s}'.format(img_data_dir, slide_name)
            if not os.path.exists(sample_folder):
                os.makedirs(sample_folder)
            # else:
            #     continue

            N_slide += 1
            print('Processing slide {:d}: {:s}'.format(N_slide, slide_name))
            slidePath = '{:s}/{:s}/{:s}'.format(svs_data_dir, folder, slide_filename)
            slide = open_slide(slidePath)

            if 'aperio.AppMag' not in slide.properties:
                print('no magnification param')
                continue
                # magnification = mag_missing_dict[slide_name]
            else:
                magnification = float(slide.properties['aperio.AppMag'])

            extract_patch_size = int(patch_size * magnification / target_mag)
            scale = 20.0 / patch_size

            w, h = slide.level_dimensions[0]

            # remove some pixels on the bottom and right edges of the image
            w = w // extract_patch_size * extract_patch_size
            h = h // extract_patch_size * extract_patch_size

            patch_mask = np.zeros((int(h*scale), int(w*scale)))  # the mask to indicate which parts are extracted

            count = 0
            num_patch = 0

            time_slide_reading = 0
            time_resize = 0
            time_write_image = 0
            time_all = 0
            for i in tqdm(range(0, w, extract_patch_size)):
                for j in range(0, h, extract_patch_size):
                    t1 = time.time()
                    patch = slide.read_region((i, j), level=0, size=[extract_patch_size, extract_patch_size])
                    count += 1

                    t2 = time.time()
                    # downsample to target patch size
                    patch = patch.resize([patch_size, patch_size])

                    t3 = time.time()
                    # check if the patch contains tissue
                    patch_gray = patch.convert('1')
                    ave_pixel_val = np.array(patch_gray).mean()
                    if ave_pixel_val < thresh_high:
                        num_patch += 1
                        patch_name = '{:s}/{:d}.png'.format(sample_folder, count)
                        patch.save(patch_name)
                        c1 = int(i*scale)
                        c2 = int((i+extract_patch_size)*scale)
                        r1 = int(j*scale)
                        r2 = int((j+extract_patch_size)*scale)
                        patch_mask[r1:r2, c1:c2] = 1

                    t4 = time.time()

                    time_slide_reading += t2 - t1
                    time_resize += t3 - t2
                    time_write_image += t4 - t3
                    time_all += t4 - t1

            print('\n\tTotal time: {:.2f}'.format(time_all))
            print('\tTime to load slides: {:.2f} ({:.2f} %)'.format(time_slide_reading, time_slide_reading/time_all * 100))
            print('\tTime to downsample patches: {:.2f} ({:.2f} %)'.format(time_resize, time_resize/time_all * 100))
            print('\tTime to write images: {:.2f} ({:.2f} %)'.format(time_write_image, time_write_image/time_all * 100))
            print('\tTotal number: {:d}'.format(num_patch))

            # create mask image folder if it doesn't exist
            mask_folder = '{:s}/mask'.format(img_data_dir)
            if not os.path.exists(mask_folder):
                os.makedirs(mask_folder)

            # save mask images
            mask_name = '{:s}/{:s}_mask.png'.format(mask_folder, slide_name)
            io.imsave(mask_name, patch_mask)


def rgb2gray(image):
    gray = np.zeros((image.shape[0], image.shape[1]))  # init 2D numpy array
    for i in range(len(image)):
        for j in range(len(image[i])):
            pixel = image[i][j]
            gray[i][j] = 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]
    return gray


if __name__ == '__main__':
    main()