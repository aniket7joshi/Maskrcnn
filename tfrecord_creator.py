import tensorflow as tf
import pandas as pd
import tempfile
import json,os,tqdm, ast
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from math import ceil,floor
from PIL import Image,ImageDraw
from io import BytesIO
import random, sys
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import cv2

_NUM_SHARDS = 4
debug = True

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _annotate_and_save_image(image_path,xmins,ymins,xmaxs,ymaxs, mask):
    print("DEBUG")
    print(image_path)
#     with Image.open(image_path) as img:
# #         img1=ImageDraw.Draw(img)
#         for i in range(len(xmins)):
#             lst=[xmins[i],ymins[i],xmaxs[i],ymaxs[i]]
# #             img1.rectangle(lst)
# #             print(lst)
#         print(np.unique(mask))
#         print(len(xmins), mask.shape)
    img = cv2.imread(image_path)
#     img1 = np.zeros((img.shape[0], img.shape[1]))
# #     print(img.shape)
    
#     for i in range(mask.shape[2]):
#         for j in range(img.shape[0]):
#             for k in range(img.shape[1]):
#                 if(mask[j,k,i]):
#                     img1[j,k] = 255
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.imshow(mask)

def annToRLE(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = ast.literal_eval(ann)
#     segm = ast.literal_eval(segm) 
#     print(type(segm))
    if isinstance(segm, list):
#         print('if' + str(segm))
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
#         print(segm)
#         print(height, width)
        rles = maskUtils.frPyObjects(segm, height, width)
        rle = maskUtils.merge(rles)
    elif isinstance(segm['counts'], list):
        # uncompressed RLE
#         print('elif')
        rle = maskUtils.frPyObjects(segm, height, width)
    else:
        # rle
#         print('else')
        rle = ann
#     print('entered')
    return rle

def annToMask(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = annToRLE(ann, height, width)
    m = maskUtils.decode(rle)
    return m


def _create_tf_record_row(dataset_group, img_dir, fid):
    xmins=dataset_group['xmin'].tolist()
    ymins=dataset_group['ymin'].tolist()
    xmaxs=dataset_group['xmax'].tolist()
    ymaxs=dataset_group['ymax'].tolist()
    labels=dataset_group['label_id'].tolist()
    allsegs = dataset_group['segmentation'].tolist()
    height=int(dataset_group['image_height'].iloc[0])
    width=int(dataset_group['image_width'].iloc[0])
    instance_masks = []
    for i in range(len(allsegs)):
        m = annToMask(allsegs[i], int(dataset_group['image_height'].iloc[0]), int(dataset_group['image_width'].iloc[0]))
        instance_masks.append(m)
    mask = np.stack(instance_masks, axis=2).astype(np.bool)
    mask = np.array(mask)
#     print(mask.shape)
#     mask_x = np.argmax(mask, axis = -1)
#     mask_x = mask_x.astype('uint8')
#     mask_img = Image.fromarray(mask_x)
#     mask_img1 = np.asarray(mask_img)
#     byts=BytesIO()
#     mask_img.save(byts,format='jpeg')
#     mask_string=byts.getvalue()


    mask_bytes = mask.tobytes()
#     new_mask = np.frombuffer(mask_bytes, dtype = 'bool')
#     new_mask = new_mask.reshape((height, width, len(labels)))    
    image_shape=None
    image_string=None

    image_file=os.path.join(img_dir,dataset_group['image_name'].iloc[0])
#     if random.random()>0.9 and debug:
#     _annotate_and_save_image(image_file,xmins,ymins,xmaxs,ymaxs, mask_img)
    image_id=int(dataset_group['image_id'].iloc[0])
    assert image_file.endswith(('.jpg','.jpeg','.png')),'required `.jpg or .jpeg or .png ` image got instead {}'.format(image_file)
#     print(image_file)
    if image_file.endswith('.png') :
        img=Image.open(image_file+str())
        if img.mode!='RGB':
            img=img.convert('RGB')
        image_file=BytesIO()
        img.save(image_file,format='jpeg')

    with Image.open(image_file) as img:
        if img.mode!='RGB':
            img=img.convert('RGB')
            byts=BytesIO()
            img.save(byts,format='jpeg')
            image_string=byts.getvalue()
        else:
            if type(image_file) is str: 
              with open(image_file,'rb') as imgfile:
                image_string = imgfile.read()
            else:
                image_string=image_file.getvalue()
        image_shape=img.size
#     print(image_shape)
    
    tf_example = tf.train.Example(features=tf.train.Features(feature={
          'image/height': int64_feature(height),
          'image/width': int64_feature(width),
          'image/encoded': bytes_feature(image_string),
          'image/object/bbox/xmin': float_list_feature(xmins),
          'image/object/bbox/xmax': float_list_feature(xmaxs),
          'image/object/bbox/ymin': float_list_feature(ymins),
          'image/object/bbox/ymax': float_list_feature(ymaxs),
          'image/f_id': int64_feature(image_id),
          'image/object/class/label': int64_list_feature(labels),
          'image/object/mask': bytes_feature(mask_bytes),
          })) 
    return tf_example


def _create_tfrecords_from_dataset(dataset, img_dir, out_path, train_only=True):

    # Get size of train/test set
    dataset_groups = [df for _, df in dataset.groupby('image_id')]
    total_images = len(dataset_groups)
#     print(dataset_groups[0])
    num_per_shard = int(ceil(total_images / _NUM_SHARDS))

    fid = 1

    for shard_id in range(_NUM_SHARDS):
        output_filename = os.path.join(
            out_path,
            '%s-%05d-of-%05d.tfrecord' % ('train', shard_id, _NUM_SHARDS))

        with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, total_images)
            for i in range(start_idx, end_idx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                    i + 1, total_images, shard_id))
                sys.stdout.flush()
                # Read the image.

                tf_example = _create_tf_record_row(dataset_groups[i], img_dir, fid)
                tfrecord_writer.write(tf_example.SerializeToString())
                fid += 1
    sys.stdout.write('\n')
    sys.stdout.flush()

    return 0



def create_tfrecords(meta_json_path, dataset_csv_path, out_path='/tmp/'):

    dataset_df=pd.read_csv(dataset_csv_path)
    with open(meta_json_path,'r') as json_file:
        meta_info=json.loads(json_file.read())
#     print(meta_info)
    train_dataset = dataset_df[dataset_df['train_only']==True]
    train_img_dir = meta_info['train_image_dir']
    meta_info['num_training_images'] = _create_tfrecords_from_dataset(train_dataset,train_img_dir,train_only=True,out_path=out_path)
    
    train_dataset = dataset_df[dataset_df['train_only']==False]
    if len(train_dataset)>0:
        val_img_dir = meta_info['val_image_dir']
        if val_img_dir is None:
            val_img_dir = train_img_dir
        meta_info['num_test_images'] = _create_tfrecords_from_dataset(train_dataset,val_img_dir,train_only=False,out_path=out_path)
    else:
        meta_info['num_test_images'] = 0


    meta_info['tfrecord_path'] = out_path
#     print(meta_info)
    return meta_info
