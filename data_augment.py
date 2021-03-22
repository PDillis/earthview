import os
import sys

import argparse
import glob
from tqdm import tqdm
import cv2


# ===========================================================================================


def multi_crop_local_images(n_crops=3):
    # TODO: detect that n_crops is None, so do the crop calculation automatically (define another function inside)
    images_path = os.path.join('datasets', 'earth_view', 'full_resolution')  # TODO: datasets/earth_view is constant
    save_path = os.path.join('datasets', 'earth_view', 'triple_cropped')  # TODO: datasets/earth_view is constant, n_crops should matter
    full_res_img_paths = glob.glob(os.path.join(images_path, '*.jpg'))

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
            print(f'Image "{img_path}" is square, will skip!')
            continue

    # Sanity check: number of cropped images is 3x original length
    cropped_image_paths = glob.glob(os.path.join(save_path, '*.jpg'))
    diff = 3 * len(full_res_img_paths) - len(cropped_image_paths)
    assert diff == 0, f'Something went wrong, missing {diff} images in {save_path}!'


# ===========================================================================================


def resize_local_images(target_size=1024):
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


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    parser_crop_images = subparsers.add_parser('crop-images', help='Crop the images')
    parser_crop_images.set_defaults(func=multi_crop_local_images)

    parser_resize_images = subparsers.add_parser('resize-images', help='Resize the images')
    parser_resize_images.set_defaults(func=resize_local_images)

    args = parser.parse_args()
    kwargs = vars(args)
    subcmd = kwargs.pop('command')

    if subcmd is None:
        print('Error: missing subcommand. Re-run with --help for usage.')
        sys.exit(1)

    func = kwargs.pop('func')
    func(**kwargs)
