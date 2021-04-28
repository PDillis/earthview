import os
import sys
from typing import Union

import argparse
import click
import glob
from tqdm import tqdm

import cv2
import PIL.Image
import numpy as np

from download_images import make_zip_file


accepted_filetypes = ('.jpg', '.jpeg', '.png')

# ===========================================================================================


@click.group()
def main():
    pass


# ===========================================================================================


@main.command(name='cut-crop')
def cut_crop_local_images(target_size: int = 1024, n_crops: int = 3) -> None:
    # TODO: detect that n_crops is None, so do the crop calculation automatically (define another function inside)
    images_path = os.path.join('images', 'all', 'full_resolution')
    save_path = os.path.join('images', 'all', 'multi_cropped', f'{target_size}')

    full_res_img_paths = []
    for root, _, files in os.walk(images_path):
        for file in files:
            if file.endswith(accepted_filetypes):
                full_res_img_paths.append(os.path.join(root, file))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for img_path in tqdm(full_res_img_paths, desc='Cropping images...', unit='images'):
        img_base_name = os.path.basename(img_path)
        img_name = os.path.splitext(img_base_name)[0]

        img = cv2.imread(img_path)
        h, w, c = img.shape
        # Wide image, so move from left to right
        if w > h:
            step_size = (w - h) // n_crops
            for i in range(n_crops):
                new_img = img[:, i * step_size: h + i * step_size, :]
                cv2.imwrite(os.path.join(save_path, f'{img_name}_{i}.jpg'), new_img)
        # Tall image, so move from top to bottom
        elif h > w:
            step_size = (h - w) // n_crops
            for i in range(n_crops):
                new_img = img[i * step_size: w + i * step_size, :, :]
                cv2.imwrite(os.path.join(save_path, f'{img_name}_{i}.jpg'), new_img)
        else:
            # Image is square, so skip
            continue

    # Sanity check: number of cropped images is 3x original length
    cropped_image_paths = glob.glob(os.path.join(save_path, '*.jpg'))
    diff = 3 * len(full_res_img_paths) - len(cropped_image_paths)
    assert diff == 0, f'Something went wrong, missing {diff} images in {save_path}!'


# ===========================================================================================


@main.command(name='resize')
def resize_local_images(target_size: int = 1024) -> None:
    """
    Resize all the local images to a desired target size.
    :param target_size: Target width and height of the square image
    :return: Images will be resized to the desired size
    """
    images_paths = os.path.join('datasets', 'earth_view', 'triple_cropped')
    save_path = os.path.join('datasets', 'earth_view', 'resized', f'{target_size}')
    cropped_images_paths = glob.glob(os.path.join(images_paths, '*.jpg'))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for img_path in tqdm(cropped_images_paths, desc='Resizing images...', unit='images'):
        img_name = os.path.basename(img_path)
        img_resized_path = os.path.join(save_path, f'{img_name}_resized{target_size}.jpg')
        # Sanity check: skip if resized image already exists
        if cv2.haveImageReader(img_resized_path):
            continue
        img = cv2.imread(img_path)
        # Sanity check: make sure it's a square image
        h, w, c = img.shape
        if h != w:
            # Skip, but leave a trail
            print(f'"{img_path}" not a square image! Shape: ({h}, {w}, {c})')
            continue
        # Pass: yay, so we resize and save it
        img_resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(img_resized_path, img_resized)

    # Sanity check: same number of resized as original images
    resized_images_paths = glob.glob(os.path.join(save_path, '*.jpg'))
    diff = len(cropped_images_paths) - len(resized_images_paths)
    assert diff == 0, f'Something went wrong, missing {diff} images in {save_path}!'


# ===========================================================================================


@main.command(name='multi-crop')
@click.option('--target-size', '-size', type=int, help='Size of squares to crop out of full res images', default=1024, show_default=True)
@click.option('--fullres-path', '-fp', type=click.Path(), help='Path to the full resolution images', default=os.path.join(os.getcwd(), 'images'))
@click.option('--img-save-path', '-sp', type=click.Path(), help='Path to save the cropped images', default=os.path.join(os.getcwd(), 'images'))
@click.option('--make-zip', '-z', is_flag=True, help='Make ZIP file with all the cropped images (easier to move around)')
def multi_crop_local_images(
        target_size: int,
        fullres_path: Union[str, os.PathLike],
        img_save_path: Union[str, os.PathLike],
        make_zip: bool) -> None:
    """
    Reproduction of multi-cropping an image (used in the BreCaHAD dataset in StyleGAN2-ADA)
        https://github.com/NVlabs/stylegan2-ada/blob/1ea5f6fa58108ca9fb94140320a1cdf515c1e246/dataset_tool.py#L836

    However, they use a static overlap between the images, which doesn't translate well to all datasets (where
    individual images may have different dimensions). This code then will try to automate this overlap with the
    desired target size. Note this isn't meant for a conditional dataset!

    :param target_size: The size of the crops; if using for a vanilla StyleGAN1/2/2-ADA, make sure it's a power of 2
    :param fullres_path: Path to the full-resolution images
    :param img_save_path: Root path where we will save the images at
    :param make_zip: Make a ZIP file with all the images; to be saved at './images/zip_files'
    :return: Images will be saved at the specified path in target_sizextarget_size resolution
    """
    # Set the final save path for the images
    save_path = os.path.join(img_save_path, 'all', 'multi_cropped', f'{target_size}')

    # Get all the path images
    full_res_img_paths = []
    for root, _, files in os.walk(os.path.join(fullres_path, 'all', 'full_resolution')):
        for file in files:
            if file.endswith(accepted_filetypes):
                full_res_img_paths.append(os.path.join(root, file))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # We go through each image, cutting it according to the dimensions and target_size
    # TODO: optimize this loop with multithreading
    for img_path in tqdm(full_res_img_paths, desc='Cropping images...', unit='images'):
        img_base_name = os.path.basename(img_path)  # 'images/all/full_resolution/1003.jpg' -> '1003.jpg'
        # We will use the image name (here, a number) and the image format (.jpg)
        img_name, ext = os.path.splitext(img_base_name)  # '1003.jpg' -> ('1003', '.jpg')

        # Open image and get dimensions
        img = PIL.Image.open(img_path).convert('RGB')
        w, h = img.size

        # Skip if target size is larger than either side
        if all(target_size > i for i in (h, w)):
            continue

        # Number of columns and rows to crop (guard against edge case where w or h == target_size)
        crop_cols = int(np.rint(w / target_size)) if w / target_size > 1 else 0
        crop_rows = int(np.rint(h / target_size)) if h / target_size > 1 else 0

        # Size of step to take when moving column and row-wise
        width_step = int((w - target_size) / crop_cols) if crop_cols != 0 else 0
        height_step = int((h - target_size) / crop_rows) if crop_rows != 0 else 0

        # Get all the crops
        for i in range(crop_cols + 1):
            for j in range(crop_rows + 1):
                # Keep the original name, but add the cropped number (easier to differentiate)
                save_name = os.path.join(save_path, f'{img_name}_cropped{2*i + j}{ext}')  # _cropped{0,1,2,...}
                # If image exists, open it, and if there's no error, skip (useful if restarting)
                if cv2.haveImageReader(save_name):
                    continue

                # Crop and save it
                new_img = img.crop((i*width_step, j*height_step,  # upper-left corner
                                    i*width_step + target_size, j*height_step + target_size))  # lower-right corner
                new_img.save(save_name)

    # Zip if desired
    if make_zip:
        print(f'Making ZIP file...')
        make_zip_file(
            parent_path_to_zip=os.path.join(img_save_path, 'all', 'multi_cropped'),
            folder_to_zip=f'{target_size}',
            zip_filename=f'all_imgs_multi-cropped{target_size}',
            path_to_save_zip=os.path.join(os.getcwd(), 'images', 'zip_files'))


# ===========================================================================================


if __name__ == '__main__':
    main()


# ===========================================================================================
