import os
import openslide
from openslide import open_slide
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


root_data_dir = '../data/TCGA-BRCA'
svs_data_dir = '{:s}/slides'.format(root_data_dir)

folder_list = os.listdir(svs_data_dir)

slides_date = {}
for folder in tqdm(folder_list):
    file_list = os.listdir('{:s}/{:s}'.format(svs_data_dir, folder))

    for filename in file_list:
        if filename[-3:] != 'svs':
            continue

        slidePath = '{:s}/{:s}/{:s}'.format(svs_data_dir, folder, filename)
        slide = open_slide(slidePath)

        slide_name = filename.split('.')[0]
        if 'aperio.Date' in slide.properties:
            slides_date[slide_name] = slide.properties['aperio.Date']
        else:
            slides_date[slide_name] = 'nan'

np.save('{:s}/slides_date.npy'.format(root_data_dir), slides_date)
with open('{:s}/slide_date.txt'.format(root_data_dir), 'w') as file:
    for k, v in sorted(slides_date.items()):
        file.write('{:s}\t{:s}\n'.format(k, v))


# all_years = []
# N = 0
# for k, v in tqdm(slides_date.items()):
#     N = N + 1
#     # if v != 'nan' and k.split('-')[1] != '3C':
#     if v != 'nan':
#         year = int(v.split('/')[-1])
#         all_years.append(year)
#
#
# n, bins, patches = plt.hist(x=all_years, bins=[7,8,9,10,11,12,13,14,15,16,17], color='#0504aa',
#                             alpha=0.7, rwidth=0.85)
# plt.grid(axis='y', alpha=0.75)
# plt.xlabel('year')
# plt.ylabel('Frequency')
# maxfreq = n.max()
# # Set a clean upper y-axis limit.
# plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
# plt.show()