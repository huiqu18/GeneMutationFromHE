import os
from openslide import open_slide
import pandas as pd


def main():
    dataset = 'TCGA-BRCA'
    root_data_dir = '../data/{:s}'.format(dataset)
    svs_data_dir = '../data/{:s}/slides'.format(dataset)

    # # load slide filenames
    # slides_filenames = sorted(os.listdir(svs_data_dir))
    # N = len(slides_filenames)

    slides_filenames = pd.read_csv('{:s}/slide_selection_final.txt'.format(root_data_dir), header=None)
    slides_filenames = list(slides_filenames[0])
    N = len(slides_filenames)
    print(N)

    slide_size_info = pd.DataFrame(columns=['Slide_name', 'Mag', 'Height', 'Width'])

    folders = os.listdir(svs_data_dir)
    for folder in folders:
        files = os.listdir('{:s}/{:s}'.format(svs_data_dir, folder))
        for filename in files:
            if filename[-3:] != 'svs':
                continue
            slide_name = filename.split('.')[0]
            if slide_name not in slides_filenames:
                continue

            slidePath = '{:s}/{:s}/{:s}'.format(svs_data_dir, folder, filename)
            slide = open_slide(slidePath)

            if 'aperio.AppMag' not in slide.properties:
                magnification = -1
                print(slide_name)
            else:
                magnification = float(slide.properties['aperio.AppMag'])
            w, h = slide.level_dimensions[0]
            slide_size_info = slide_size_info.append({'Slide_name': slide_name, 'Mag': int(magnification),
                                                      'Height': int(h), 'Width': int(w)}, ignore_index=True)

    slide_size_info.to_pickle('{:s}/slide_size_info.pickle'.format(root_data_dir))
    print(slide_size_info.shape)


if __name__ == '__main__':
    main()