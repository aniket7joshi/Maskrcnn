"""Summary
"""
import tensorflow as tf
import cv2, os, glob
import numpy as np
# from .preprocessing import anchor_targets_bbox, anchors_for_shape

def parse_tfrecords(
  filenames, 
  batch_size, 
  num_classes,
  preprocess_fn=lambda x:x):
    """Summary
    
    Args:
        filenames (TYPE): Description
        batch_size (TYPE): Description
        min_side (int, optional): Description
        max_side (int, optional): Description
    
    Returns:
        tf.data.Dataset: Description
    """
    def pad_resize(image, height, width):
        """Summary
        
        Args:
            image (TYPE): Description
            height (TYPE): Description
            width (TYPE): Description
            scale (TYPE): Description
        
        Returns:
            numpy nd.array: Description
        """
        padded_image = np.zeros(shape=(height.astype(int), width.astype(int),3), dtype=image.dtype)
        h,w,_ =  image.shape
        padded_image[:h,:w,:] = image
#         resized_image = cv2.resize(padded_image, None).astype(tf.keras.backend.floatx())
        return preprocess_fn(padded_image)

    @tf.function
    def decode_pad_resize(image_string, pad_height, pad_width, height, width):
      """Summary
      
      Args:
          image_string (TYPE): Description
          pad_height (TYPE): Description
          pad_width (TYPE): Description
          scale (TYPE): Description
      
      Returns:
          tf.tensor: Description
      """
      image = tf.image.decode_jpeg(image_string)
      print(image.shape, height, width)
      image = tf.numpy_function(pad_resize, [image, pad_height, pad_width], Tout=tf.uint8)
      #image.set_shape([None, None, 3])
      return image

    def process_bboxes(image_array, bboxes, labels):

        # delete bboxes containing [-1,-1,-1,-1]
        bboxes = bboxes[~np.all(bboxes==-1, axis=1)]
        # delete labels containing[-1]
        labels = labels[labels>-1]#[0]

        # generate raw anchors
        raw_anchors = anchors_for_shape(
            image_shape=image_array.shape,
            sizes=sizes,
            ratios=ratios,
            scales=scales,
            strides=strides,
            pyramid_levels=[3, 4, 5, 6, 7],
            shapes_callback=None,
        )

        # generate anchorboxes and class labels      
        gt_regression, gt_classification = anchor_targets_bbox(
              anchors=raw_anchors,
              image=image_array,
              bboxes=bboxes,
              gt_labels=labels,
              num_classes=num_classes,
              negative_overlap=0.4,
              positive_overlap=0.5
          )

        return gt_regression, gt_classification

    @tf.function
    def tf_process_bboxes(xmin_batch, ymin_batch, xmax_batch, ymax_batch, label_batch, image_batch):

        regression_batch = list()
        classification_batch = list()

        for index in range(batch_size):
            xmins, ymins, xmaxs, ymaxs, labels = xmin_batch[index], ymin_batch[index], xmax_batch[index], ymax_batch[index], label_batch[index]
            image_array = image_batch[index]
            bboxes = tf.convert_to_tensor([xmins,ymins,xmaxs,ymaxs], dtype=tf.keras.backend.floatx())
            bboxes = tf.transpose(bboxes)
            gt_regression, gt_classification = tf.numpy_function(process_bboxes, [image_array, bboxes, labels], Tout=[tf.keras.backend.floatx(), tf.keras.backend.floatx()])

            regression_batch.append(gt_regression)
            classification_batch.append(gt_classification)

        return tf.convert_to_tensor(regression_batch), tf.convert_to_tensor(classification_batch)

        #return bboxes
        

    def _parse_function(serialized):
        """Summary
        
        Args:
            serialized (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        features = {
          'image/height': tf.io.FixedLenFeature([], tf.int64),
          'image/width': tf.io.FixedLenFeature([], tf.int64),
          'image/encoded': tf.io.FixedLenFeature([],tf.string),
          'image/object/bbox/xmin': tf.io.VarLenFeature(tf.keras.backend.floatx()),
          'image/object/bbox/xmax': tf.io.VarLenFeature(tf.keras.backend.floatx()),
          'image/object/bbox/ymin': tf.io.VarLenFeature(tf.keras.backend.floatx()),
          'image/object/bbox/ymax': tf.io.VarLenFeature(tf.keras.backend.floatx()),
          'image/f_id': tf.io.FixedLenFeature([], tf.int64),
          'image/object/class/label':tf.io.VarLenFeature(tf.int64)}


        parsed_example = tf.io.parse_example(serialized=serialized, features=features)

        max_height = tf.cast(tf.keras.backend.max(parsed_example['image/height']), tf.int32)
        max_width = tf.cast(tf.keras.backend.max(parsed_example['image/width']), tf.int32)

        smallest_side = tf.keras.backend.minimum(max_height, max_width)
        # rescale the image so the smallest side is min_side
#         scale = min_side / smallest_side
        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = tf.keras.backend.maximum(max_height, max_width)

        # scale = tf.cond(largest_side * tf.cast(scale, tf.int32) > max_side, lambda: max_side / largest_side, , lambda: scale)
#         if largest_side * tf.cast(scale, tf.int32) > max_side:
#             scale = max_side / largest_side

#         scale = tf.cast(scale, tf.keras.backend.floatx()) 
        height = features['image/height']
        width = features['image/width']
        image_batch = tf.map_fn(lambda x: decode_pad_resize(x, max_height, max_width,height, width), parsed_example['image/encoded'], dtype=tf.uint8)
        #print(scale)

        xmin_batch = tf.sparse.to_dense(parsed_example['image/object/bbox/xmin'], default_value=-1)
        xmax_batch = tf.sparse.to_dense(parsed_example['image/object/bbox/xmax'], default_value=-1)
        ymin_batch = tf.sparse.to_dense(parsed_example['image/object/bbox/ymin'], default_value=-1)
        ymax_batch = tf.sparse.to_dense(parsed_example['image/object/bbox/ymax'], default_value=-1)

        annotation_batch = []

        label_batch = tf.sparse.to_dense(parsed_example['image/object/class/label'], default_value=-1)

        # tf.print(xmin_batch.shape)
        # tf.print(ymin_batch.shape)
        # tf.print(xmax_batch.shape)
        # tf.print(ymax_batch.shape)
        # tf.print(label_batch.shape)

#         regression_batch, classification_batch = tf_process_bboxes(xmin_batch, ymin_batch, xmax_batch, ymax_batch, label_batch, image_batch)

        # create GT from annotations
        # augment image_batch
        # normalize image through preprocessing


        return image_batch , {'label_batch':label_batch}


    # dataset = tf.data.Dataset.from_tensor_slices(filenames).repeat(-1)
    dataset = tf.data.Dataset.list_files(filenames).shuffle(buffer_size=256).repeat(-1)
    dataset = dataset.interleave(
      tf.data.TFRecordDataset, 
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
      deterministic=False)

    dataset = dataset.batch(
      batch_size, 
      drop_remainder=True)    # Batch Size

    dataset = dataset.map(
      _parse_function, 
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # dataset = dataset.cache()
#     dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


# if __name__ == '__main__':
#     # filepath = os.path.join(os.getcwd(),'DATA','train*.tfrecord')

#     dataset = parse_tfrecords(
#         filenames=os.path.join(os.getcwd(),'DATA','train*.tfrecord'), 
#         batch_size=2,
#         num_classes=5)

#     for data, annotation in dataset.take(10):
#         image_batch = data.numpy()

#         abxs_batch = annotation['regression'].numpy()
#         labels_batch = annotation['classification'].numpy()

#         print(image_batch.shape, abxs_batch.shape, labels_batch.shape)
#         print(image_batch.dtype, abxs_batch.dtype, labels_batch.dtype)