import os
import sys

sys.path.append('/code_location/WaveNet-Enhancement')
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
import tensorflow as tf
import glob
import collections


def build_tfrecords_iterator(num_epochs, train_path_pattern, is_bkgd, feature_parsing_dict, narrowband_noise,
                             manually_added, STIM_SIZE, localization_bin_resolution, stacked_channel):
    '''
    Builds tensorflow iterator for feeding graph with data from tfrecords.
    
    Args
    ----
    tfrecords_regex (str): regular expression capturing all tfrecords to include in dataset
    feature_parsing_dict (dict): keys are tfrecords feature keys, values are dictionaries with 'dtype' and 'shape' keys
    iterator_type (str): must be either 'one-shot' (for training) or 'initializable' (for validation)
    num_epochs (int): number of times to repeat dataset
    batch_size (int): number of examples per batch
    n_prefetch (int): argument for dataset.prefetch (max number of elements to buffer when prefetching)
    buffer (int): argument for dataset.shuffle (size of shuffle buffer)
    shuffle_flag (bool): if True, dataset will be shuffled
    
    Returns
    -------
    iterator (tf iterator object): iterator whose `get_next()` method returns `input_tensor_dict`
    dataset (tf dataset object): dataset object used to construct the iterator
    iterator_saveable_object (tf saveable object): saveable object for saving the iterator state
    '''
    ### Helper dictionary to map strings to tf.dtype objects (which are not easily saved in JSON)

    training_paths = glob.glob(train_path_pattern)

    string_to_dtype = {'tf.float32': tf.float32, 'tf.float64': tf.float64, 'tf.int32': tf.int32, 'tf.int64': tf.int64,
        'tf.string': tf.string, }

    ### Set up feature_dict to use for parsing tfrecords
    feature_dict = collections.OrderedDict()
    # feature_dict = {}
    # for path in sorted(feature_parsing_dict.keys()):
    for path in feature_parsing_dict.keys():
        if is_bkgd is True and (path == 'train/azim' or path == 'train/elev'):
            path_dtype = feature_parsing_dict[path].dtype
            feature_dict[path] = tf.VarLenFeature(path_dtype)
        else:
            path_dtype = feature_parsing_dict[path].dtype
            path_shape = feature_parsing_dict[path].shape
            feature_dict[path] = tf.FixedLenFeature(path_shape, path_dtype)

    ### Define the tfrecords parsing function
    def parse_tfrecord_example(record):
        ''' Parsing function returns dictionary of tensors with tfrecords paths as keys '''
        parsed_features = tf.parse_single_example(record, features=feature_dict)

        # Create a dictionary of input tensors
        # -> Decode the image, cast it to float32 and reshape it to the original shape
        # Nothing else really...
        input_tensor_dict = {}
        for path in feature_parsing_dict.keys():
            # ['train/azim', 'train/elev', 'train/image', 'train/image_height', 'train/image_width']
            if is_bkgd is True and (path == 'train/azim' or path == 'train/elev'):
                input_tensor_dict[path] = parsed_features[path]
            else:
                path_dtype = feature_parsing_dict[path].dtype
                if path == 'train/image':
                    if path_dtype == tf.string:
                        path_dtype = tf.float32
                    decoded_bytes_feature = tf.decode_raw(parsed_features[path], path_dtype)
                    if decoded_bytes_feature.dtype == tf.float64:
                        decoded_bytes_feature = tf.cast(decoded_bytes_feature, tf.float32)
                    input_tensor_dict[path] = tf.reshape(decoded_bytes_feature, STIM_SIZE)
                else:
                    input_tensor_dict[path] = parsed_features[path]

        # - Cast all int64 dtypes to int32
        # - Divide the azim and elev labels by 5
        # -> Why divide elevation by 5?
        # - I think: replace dtype strings with tf.dtype objects...?
        label_div_const = tf.constant([localization_bin_resolution])
        for elem in input_tensor_dict:
            if elem != 'train/image':
                if input_tensor_dict[elem].dtype == 'float32':
                    v = tf.cast(input_tensor_dict[elem], tf.float32)
                    input_tensor_dict[elem] = v
                elif input_tensor_dict[elem].dtype == 'int64':
                    if elem == 'train/azim':
                        azims = tf.cast(input_tensor_dict['train/azim'], tf.int32)
                        if not is_bkgd and (narrowband_noise or not manually_added):
                            azims = tf.div(azims, label_div_const)
                        input_tensor_dict['train/azim'] = azims
                    elif elem == 'train/elev':
                        elevs = tf.cast(input_tensor_dict['train/elev'], tf.int32)
                        if not is_bkgd and (narrowband_noise or not manually_added):
                            elevs = tf.div(elevs, label_div_const)
                        input_tensor_dict['train/elev'] = elevs
                    else:
                        v = tf.cast(input_tensor_dict[elem], tf.int32)
                        input_tensor_dict[elem] = v

        # This does nothing, right? oO
        if stacked_channel:
            images = tf.slice(input_tensor_dict['train/image'], [0, 0, 0], [39, 48000, 2])
        else:
            images = tf.slice(input_tensor_dict['train/image'], [0, 0], [78, 48000])
        input_tensor_dict['train/image'] = images
        return input_tensor_dict

    #        if is_bkgd:
    #            image = tf.reshape(image, BKGD_SIZE)
    #            if background_textures:
    #                azim = tf.cast(features['train/azim'],tf.int32)
    #                elev = tf.cast(features['train/elev'],tf.int32)
    #                return image,azim,elev
    #            if not all_positions_bkgd:
    #                label = tf.cast(features['train/label'], tf.int32)
    #                return image,label
    #            return image
    #        else:
    #            if tone_version:
    #                image = tf.reshape(image, TONE_SIZE)
    #                tone = tf.cast(features['train/freq'],tf.int32)
    #                if itd_tones:
    #                    azim = tf.cast(features['train/azim'], tf.int32)
    #                    elev = tf.cast(features['train/elev'], tf.int32)
    #                    label_div_const = tf.constant([localization_bin_resolution])
    #                    if not manually_added:
    #                        azim = tf.div(azim,label_div_const)
    #                        elev = tf.div(elev,label_div_const)
    #                    return image, azim, elev, tone
    #                else:
    #                    label = tf.cast(features['train/label'], tf.int32)
    #                    label_div_const = tf.constant([10])
    #                    label = tf.div(label,label_div_const)
    #                return image,label, tone
    #            elif sam_tones or transposed_tones:
    #                image = tf.reshape(image,STIM_SIZE)
    #                carrier_freq = tf.cast(features['train/carrier_freq'], tf.int32)
    #                modulation_freq = tf.cast(features['train/modulation_freq'], tf.int32)
    #                flipped = tf.cast(features['train/flipped'], tf.int32)
    #                if sam_tones:
    #                    carrier_delay = tf.cast(features['train/carrier_delay'], tf.float32)
    #                    modulation_delay = tf.cast(features['train/modulation_delay'], tf.float32)
    #                    return (image, carrier_freq, modulation_freq,
    #                            carrier_delay, modulation_delay, flipped)
    #                elif transposed_tones:
    #                    delay = tf.cast(features['train/delay'], tf.float32)
    #                    return (image, carrier_freq, modulation_freq,
    #                            delay, flipped)
    #            elif precedence_effect:
    #                image = tf.reshape(image,STIM_SIZE)
    #                delay = tf.cast(features['train/delay'], tf.float32)
    #                start_sample = tf.cast(features['train/start_sample'], tf.int32)
    #                lead_level = tf.cast(features['train/lead_level'], tf.float32)
    #                lag_level = tf.cast(features['train/lag_level'], tf.float32)
    #                flipped = tf.cast(features['train/flipped'], tf.int32)
    #                return (image, delay, start_sample, lead_level,
    #                        lag_level, flipped)
    #            elif narrowband_noise:
    #                image = tf.reshape(image,STIM_SIZE)
    #                azim = tf.cast(features['train/azim'], tf.int32)
    #                elev = tf.cast(features['train/elev'], tf.int32)
    #                label_div_const = tf.constant([localization_bin_resolution])
    #                azim = tf.div(azim,label_div_const)
    #                elev = tf.div(elev,label_div_const)
    #                bandwidth = tf.cast(features['train/bandwidth'], tf.float32)
    #                center_freq = tf.cast(features['train/center_freq'], tf.int32)
    #                return (image, azim, elev, bandwidth, center_freq)
    #            else:
    #                azim = tf.cast(features['train/azim'], tf.int32)
    #                elev = tf.cast(features['train/elev'], tf.int32)
    #                label_div_const = tf.constant([localization_bin_resolution])
    #                if not manually_added:
    #                    azim = tf.div(azim,label_div_const)
    #                    elev = tf.div(elev,label_div_const)
    #                image = tf.reshape(image,STIM_SIZE)
    #                # Reshape image data into the original shape
    #                if branched:
    #                    class_num = tf.cast(features['train/class_num'], tf.int32)
    #                    return image, azim, elev, class_num
    #                if freq_label:
    #                    tone = tf.cast(features['train/freq'],tf.int32)
    #                    return image, azim, elev, tone
    #                return image, azim, elev

    ### Create tensorflow dataset
    #    input_data_filenames = sorted(glob.glob(train_path_pattern))
    #    print('### Files found: {}'.format(len(input_data_filenames)))
    #    print(input_data_filenames[0],'\n...\n', input_data_filenames[-1])

    dataset_paths = tf.data.Dataset.list_files(train_path_pattern).shuffle(len(training_paths))
    #    print ("old dataset:")
    #    print (dataset)
    dataset = dataset_paths.apply(tf.contrib.data.parallel_interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP").map(parse_tfrecord_example, num_parallel_calls=1),
        cycle_length=10, block_length=16))
    #    print (dataset)
    dataset = dataset.shuffle(buffer_size=200)
    if not is_bkgd:
        dataset = dataset.repeat(num_epochs)
    else:
        dataset = dataset.repeat()
    dataset = dataset.prefetch(100)

    return dataset
