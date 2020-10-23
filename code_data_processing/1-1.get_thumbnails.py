import os
from openslide import open_slide


mag_missing_dict = {'TCGA-OL-A5RU-01Z-00-DX1': 20, 'TCGA-OL-A5RV-01Z-00-DX1': 20, 'TCGA-OL-A5RW-01Z-00-DX1': 20,
                    'TCGA-OL-A5RX-01Z-00-DX1': 20, 'TCGA-OL-A5RY-01Z-00-DX1': 20, 'TCGA-OL-A5RZ-01Z-00-DX1': 20,
                    'TCGA-OL-A5S0-01Z-00-DX1': 20}

def main():
    dataset = 'TCGA-BRCA'
    svs_data_dir = '../data/{:s}/slides'.format(dataset)
    save_dir = '../data/{:s}/thumbnails'.format(dataset)
    os.makedirs(save_dir, exist_ok=True)

    patch_size = 512  # the size of patches extracted from 20x svs image
    target_mag = 20

    # load slide filenames
    folderlist = sorted(os.listdir(svs_data_dir))

    N_slide = 0
    for folder in folderlist:
        filelist = os.listdir('{:s}/{:s}'.format(svs_data_dir, folder))
        for filename in filelist:
            if filename[-3:] != 'svs':
                continue
            slide_name = filename.split('.')[0]
            N_slide += 1

            # if slide_name != 'TCGA-93-A4JQ-01Z-00-DX1':
            #     continue

            print('Processing slide {:d}: {:s}'.format(N_slide, slide_name))
            slidePath = '{:s}/{:s}/{:s}'.format(svs_data_dir, folder, filename)
            slide = open_slide(slidePath)

            if 'aperio.AppMag' not in slide.properties:
                if slide_name in mag_missing_dict:
                    magnification = mag_missing_dict[slide_name]
                else:
                    print('no magnification param')
                    continue
            else:
                magnification = float(slide.properties['aperio.AppMag'])

            extract_patch_size = int(patch_size * magnification / target_mag)
            w, h = slide.level_dimensions[0]

            th_w = int(w / extract_patch_size * 10)
            th_h = int(h / extract_patch_size * 10)
            thumbnail = slide.get_thumbnail((th_w, th_h))
            thumbnail_name = '{:s}/{:s}_thumbnail.png'.format(save_dir, slide_name)
            thumbnail.save(thumbnail_name)


if __name__ == '__main__':
    main()