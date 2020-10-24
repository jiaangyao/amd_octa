import numpy as np
from matplotlib import pyplot as plt
import os
from config.load_config import get_config
from utils.io_funcs import load_model
from pathlib import Path
import tensorflow as tf
from tf_keras_vis.utils import num_of_gpus
from model import get_model

from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from skimage import io, transform, color

from tf_keras_vis.utils import normalize

from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam

_, gpus = num_of_gpus()
print('{} GPUs'.format(gpus))

f_fig = Path('/home/jyao/Downloads/fig/')

cfg = get_config(filename=Path(os.getcwd()) / 'config' / 'default_config.yml')
cfg.d_model = f_fig
cfg.lr = 5e-5
cfg.lam = 1e-5
cfg.num_classes = 3
cfg.sample_size = [[256, 256, 5, 1], [256, 256, 1]]

angio_model = get_model('arch_009', cfg)

bscan_model = tf.keras.models.load_model(str(f_fig / 'bscanNewDataSmall_101920.h5'))
combined_model = tf.keras.models.load_model(str(f_fig / 'combinedBest_102220.h5'))
angio_model.load_weights(str(f_fig / 'arch_009_20201022_145518'))

struct_model = get_model('arch_010', cfg)
struct_model.load_weights(str(f_fig / 'arch_010_20201022_144636'))


# Image titles
image_titles = ['FN_trueCNV_predDryAMD', 'FP_trueDryAMD_predCNV']

# target_size = (224, 224)
target_size = (256, 256)

# load images from sub35

# Note that Sub 35 has true label 1 and sub 77 has true label 2

img_cube_angio_sub35 = np.zeros((target_size[0], target_size[1], 5, 1))
for i in range(5):
    curr_angio_img = load_img(str(f_fig / '35_DryAMD_FP_CNV_angio_{}.bmp'.format(i)), target_size=target_size)
    curr_angio_img = color.rgb2gray(img_to_array(curr_angio_img))
    img_cube_angio_sub35[:, :, i, :] = img_to_array(curr_angio_img).reshape(target_size[0], target_size[1], 1)

img_cube_struct_sub35 = np.zeros((target_size[0], target_size[1], 5, 1))
for i in range(5):
    curr_struct_img = load_img(str(f_fig / '35_DryAMD_FP_CNV_struct_{}.bmp'.format(i)), target_size=target_size)
    curr_struct_img = color.rgb2gray(img_to_array(curr_struct_img))
    img_cube_struct_sub35[:, :, i, :] = img_to_array(curr_struct_img).reshape(target_size[0], target_size[1], 1)

img_bscan_sub35 = load_img(str(f_fig / '35_DryAMD_FP_CNV_bscan.bmp'), target_size=target_size)
img_bscan_sub35 = color.rgb2gray(img_to_array(img_bscan_sub35)).reshape(target_size[0], target_size[1], 1)

print("Sub 35 shapes")
print(img_cube_angio_sub35.shape)
print(img_cube_struct_sub35.shape)
print(img_bscan_sub35.shape)

# load images from sub 77
img_cube_angio_sub77 = np.zeros((target_size[0], target_size[1], 5, 1))
for i in range(5):
    curr_angio_img = load_img(str(f_fig / '77_CNV_FN_DryAMD_angio_{}.bmp'.format(i)), target_size=target_size)
    curr_angio_img = color.rgb2gray(img_to_array(curr_angio_img))
    img_cube_angio_sub77[:, :, i, :] = img_to_array(curr_angio_img).reshape(target_size[0], target_size[1], 1)

img_cube_struct_sub77 = np.zeros((target_size[0], target_size[1], 5, 1))
for i in range(5):
    curr_struct_img = load_img(str(f_fig / '77_CNV_FN_DryAMD_struct_{}.bmp'.format(i)), target_size=target_size)
    curr_struct_img = color.rgb2gray(img_to_array(curr_struct_img))
    img_cube_struct_sub77[:, :, i, :] = img_to_array(curr_struct_img).reshape(target_size[0], target_size[1], 1)

img_bscan_sub77 = load_img(str(f_fig / '77_CNV_FN_DryAMD_bscan.bmp'), target_size=target_size)
img_bscan_sub77 = color.rgb2gray(img_to_array(img_bscan_sub77)).reshape(target_size[0], target_size[1], 1)

print("\nSub 77 shapes")
print(img_cube_angio_sub77.shape)
print(img_cube_struct_sub77.shape)
print(img_bscan_sub77.shape)

x_angio = np.stack([img_cube_angio_sub35, img_cube_angio_sub77], axis=0)
x_struct = np.stack([img_cube_struct_sub35, img_cube_struct_sub77], axis=0)
x_bscan = np.stack([img_bscan_sub35, img_bscan_sub77], axis=0)

print("\nFinal shapes")
print(x_angio.shape)
print(x_struct.shape)
print(x_bscan.shape)

x = [x_angio, x_struct, x_bscan]


def loss(output):
    # 1 is the imagenet index corresponding to Goldfish, 294 to Bear and 413 to Assault Rifle.
    return (output[0][1], output[1][2])


def model_modifier(m):
    m.layers[-1].activation = tf.keras.activations.linear
    return m


gradcam_angio = Gradcam(angio_model, model_modifier=model_modifier, clone=False)

# Generate heatmap with GradCAM
cam_angio = gradcam_angio(loss, x, penultimate_layer=13, expand_cam=True)
cam_angio = normalize(cam_angio[0])


# image_titles = ['Deep', 'Avascular', 'ORCC', 'Choriocapillaris', 'Choroid']
# subplot_args = { 'nrows': 1, 'ncols': 5, 'figsize': (9, 3),
#                  'subplot_kw': {'xticks': [], 'yticks': []} }
# f, ax = plt.subplots(**subplot_args)
# for i, title in enumerate(image_titles):
#     heatmap = np.uint8(cm.jet(cam_angio)[0, :, :, i] * 255)
#     ax[i].set_title(title, fontsize=14)
#     ax[i].imshow(x_angio[0, :, :, i, :])
#     ax[i].imshow(heatmap, cmap='jet', alpha=0.5) # overlay
# plt.tight_layout()
# plt.show()

gradcam_struct = Gradcam(struct_model, model_modifier=model_modifier, clone=False)

# Generate heatmap with GradCAM
cam_struct = gradcam_struct(loss, x, penultimate_layer=36, expand_cam=True)
cam_struct = normalize(cam_struct[0])

image_titles = ['Deep', 'Avascular', 'ORCC', 'Choriocapillaris', 'Choroid']
subplot_args = { 'nrows': 1, 'ncols': 5, 'figsize': (9, 3),
                 'subplot_kw': {'xticks': [], 'yticks': []} }
f, ax = plt.subplots(**subplot_args)
for i, title in enumerate(image_titles):
    heatmap = np.uint8(cm.jet(cam_struct)[0, :, :, i] * 255)
    ax[i].set_title(title, fontsize=14)
    ax[i].imshow(x_angio[0, :, :, i, :])
    ax[i].imshow(heatmap, cmap='jet', alpha=0.5) # overlay
plt.tight_layout()
plt.show()

print('nothing')

# gradcam_combined = Gradcam(combined_model, model_modifier=model_modifier, clone=False)
#
# # Generate heatmap with GradCAM
# cam_combined = gradcam_combined(loss, x, penultimate_layer=56, expand_cam=True)
# cam_combined = normalize(cam_combined)

# # Create Gradcam object
# gradcam_bscan = Gradcam(bscan_model,
#                   model_modifier=model_modifier,
#                   clone=False)
#
# # Generate heatmap with GradCAM
# cam_bscan = gradcam_bscan(loss,
#               x_bscan,
#               penultimate_layer=-1, # model.layers number
#              )
# cam_bscan = normalize(cam_bscan)
#
# subplot_args = { 'nrows': 1, 'ncols': 2, 'figsize': (9, 3),
#                  'subplot_kw': {'xticks': [], 'yticks': []} }
# f, ax = plt.subplots(**subplot_args)
# for i, title in enumerate(image_titles):
#     heatmap = np.uint8(cm.jet(cam_bscan[i])[..., :3] * 255)
#     ax[i].set_title(title, fontsize=14)
#     ax[i].imshow(x_bscan[i, :, :, :])
#     ax[i].imshow(heatmap, cmap='jet', alpha=0.5) # overlay
# plt.tight_layout()
# plt.show()
print('nothing')