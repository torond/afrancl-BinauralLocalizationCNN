import os

os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
from math import sqrt
import numpy as np
import tensorflow as tf
import glob
import time
import json
from NetBuilder_valid_pad import NetBuilder
from tfrecords_iterator import build_tfrecords_iterator
from google.protobuf.json_format import MessageToJson
from parse_nested_dictionary import parse_nested_dictionary
import collections
import scipy.signal as signallib

import memory_saving_gradients
from tensorflow.python.ops import gradients


# import mem_util

# monkey patch tf.gradients to point to our custom version, with automatic checkpoint selection
def gradients_memory(ys, xs, grad_ys=None, **kwargs):
    return memory_saving_gradients.gradients(ys, xs, grad_ys, checkpoints='memory', **kwargs)


gradients.__dict__["gradients"] = memory_saving_gradients.gradients_speed


def write_batch_data(newpath, train_path_pattern, stim, batch_acc, batch_conditional, eval_keys, counter):
    if train_path_pattern.split("/")[-2] == 'testset':
        stimuli_name = 'testset_' + train_path_pattern.split("/")[-3]
    else:
        stimuli_name = train_path_pattern.split("/")[-2]
    np.save(newpath + '/plot_array_padded_{}_count{}_iter{}.npy'.format(stimuli_name, counter, stim), batch_acc)
    np.save(newpath + '/batch_conditional_{}_count_{}_iter{}.npy'.format(stimuli_name, counter, stim),
            batch_conditional)
    acc_corr = [pred[0] for pred in batch_acc]
    acc_accuracy = sum(acc_corr) / len(acc_corr)
    with open(newpath + '/accuracies_test_{}_count{}_iter{}.json'.format(stimuli_name, counter, stim), 'w') as f:
        json.dump(acc_accuracy, f)
    with open(newpath + '/keys_test_{}_iter{}.json'.format(stimuli_name, stim), 'w') as f:
        json.dump(eval_keys, f)


def tf_record_CNN_spherical():
    # TODO
    #  - Populate parameters and remove from signature
    #  - change dataset loading to only use foreground data, i.e. disable combining
    #  - git clone my fork on the lab PC, load test data and net_weights to where they should be

    # Ex: /om/scratch/Sat/francl/bkgdRecords_textures_sparse_sampled_same_texture_expanded_set_44.1kHz_stackedCH_upsampled_anechoic/train*.tfrecords
    train_path_pattern = '/home/neurobio/PycharmProjects/BinauralLocalizationCNN/examples/training_data/training-data_orig_mcdermott.tfrecord'
    # bkgd_train_path_pattern = None
    arch_ID = 1
    model_path = '/home/neurobio/PycharmProjects/BinauralLocalizationCNN/examples/net_weights'
    newpath = model_path + '/net1'
    config_array = np.load(newpath + '/config_array.npy')

    # bkgd_training_paths = glob.glob(bkgd_train_path_pattern)
    training_paths = glob.glob(train_path_pattern)


    ###Do not change parameters below unless altering network###


    STIM_SIZE = [39, 48000, 2]
    BKGD_SIZE = [39, 48000, 2]
    n_classes_localization = 504
    n_classes_recognition = 780
    localization_bin_resolution = 5

    # Optimization Params
    batch_size = 16
    learning_rate = 1e-3
    # Change for network precision,must match input data type
    filter_dtype = tf.float32
    padding = 'VALID'

    # Downsampling Params
    sr = 48000
    cochleagram_sr = 8000
    post_rectify = True

    # Display interval training statistics
    display_step = 25
    # Changes how often data is saved to numpy arrays when dataset is large
    write_step = 15625  # 250k examples

    dropout_training_state = False
    training_state = False
    num_epochs = 1

    # No idea if this is important
    # Using these values because 5/40 are the standard training SNRs
    SNR_max = 40
    SNR_min = 5
    if not (SNR_min > 30 or SNR_max > 40):
        SNR_min = 30.0
        SNR_max = 35.0
    print("Testing SNR(dB): Max: " + str(SNR_max) + "Min: " + str(SNR_min))


    with tf.device("/cpu:0"):
        def rms(wav):
            square = tf.square(wav)
            mean_val = tf.reduce_mean(square)
            return tf.sqrt(mean_val)

        def combine_signal_and_noise_stacked_channel(signals, backgrounds, delay, sr, cochleagram_sr, post_rectify):
            tensor_dict_fg = {}
            tensor_dict_bkgd = {}
            tensor_dict = {}
            snr = tf.random_uniform([], minval=SNR_min, maxval=SNR_max, name="snr_gen")
            for path1 in backgrounds:
                if path1 == 'train/image':
                    background = backgrounds['train/image']
                else:
                    tensor_dict_bkgd[path1] = backgrounds[path1]
            for path in signals:
                if path == 'train/image':
                    signal = signals['train/image']
                    sig_len = signal.shape[1] - delay
                    sig = tf.slice(signal, [0, 0, 0], [39, sig_len, 2])
                    max_val = tf.reduce_max(sig)
                    sig_rms = rms(tf.reduce_sum(sig, [0, 2]))
                    sig = tf.div(sig, sig_rms)
                    # sig = tf.Print(sig, [tf.reduce_max(sig)],message="\nMax SIG:")
                    sf = tf.pow(tf.constant(10, dtype=tf.float32), tf.div(snr, tf.constant(20, dtype=tf.float32)))
                    bak_rms = rms(tf.reduce_sum(background, [0, 2]))
                    # bak_rms = tf.Print(bak_rms, [tf.reduce_max(bak_rms)],message="\nNoise RMS:")
                    sig_rms = rms(tf.reduce_sum(sig, [0, 2]))
                    scaling_factor = tf.div(tf.div(sig_rms, bak_rms), sf)
                    # scaling_factor = tf.Print(scaling_factor, [scaling_factor],message="\nScaling Factor:")
                    noise = tf.scalar_mul(scaling_factor, background)
                    # noise = tf.Print(noise, [tf.reduce_max(noise)],message="\nMax Noise:")
                    front = tf.slice(noise, [0, 0, 0], [39, delay, 2])
                    middle = tf.slice(noise, [0, delay, 0], [39, sig_len, 2])
                    end = tf.slice(noise, [0, (delay + int(sig_len)), 0], [39, -1, 2])
                    middle_added = tf.add(middle, sig)
                    new_sig = tf.concat([front, middle_added, end], 1)
                    # new_sig = sig
                    rescale_factor = tf.div(max_val, tf.reduce_max(new_sig))
                    # rescale_factor = tf.Print(rescale_factor, [rescale_factor],message="\nRescaling Factor:")
                    new_sig = tf.scalar_mul(rescale_factor, new_sig)
                    new_sig_rectified = tf.nn.relu(new_sig)
                    new_sig_reshaped = tf.reshape(new_sig_rectified, [39, 48000, 2])
                    # new_sig_reshaped = tf.reshape(new_sig,[72,30000,1])
                    # return (signal, background,noise,new_sig_reshaped)
                    tensor_dict_fg[path] = new_sig_reshaped
                else:
                    tensor_dict_fg[path] = signals[path]
            tensor_dict[0] = tensor_dict_fg
            tensor_dict[1] = tensor_dict_bkgd
            return tensor_dict



        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        is_bkgd = False
        first = training_paths[0]
        for example in tf.python_io.tf_record_iterator(first, options=options):
            break

        jsonMessage = MessageToJson(tf.train.Example.FromString(example))
        jsdict = json.loads(jsonMessage)
        feature = parse_nested_dictionary(jsdict, is_bkgd)

        dataset = build_tfrecords_iterator(num_epochs, train_path_pattern, is_bkgd, feature, False,
                                           False, STIM_SIZE, localization_bin_resolution, True)
        new_dataset = build_tfrecords_iterator(num_epochs, train_path_pattern, is_bkgd, feature, False,
                                           False, STIM_SIZE, localization_bin_resolution, True)

        ###READING QUEUE MACHINERY###
        ### KEEP FOR BACKGROUND COMBINING!!! ###
        #
        # # Create a list of filenames and pass it to a queue
        # bkgd_filename_queue = tf.train.string_input_producer(bkgd_training_paths, shuffle=True,
        #                                                      capacity=len(bkgd_training_paths))
        # # Define a reader and read the next record
        # options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        # bkgd_reader = tf.TFRecordReader(options=options)
        # _, bkgd_serialized_example = bkgd_reader.read(bkgd_filename_queue)
        #
        # is_bkgd = True
        # bkgd_first = bkgd_training_paths[0]
        # for bkgd_example in tf.python_io.tf_record_iterator(bkgd_first, options=options):
        #     break
        #
        # bkgd_jsonMessage = MessageToJson(tf.train.Example.FromString(bkgd_example))
        # bkgd_jsdict = json.loads(bkgd_jsonMessage)
        # bkgd_feature = parse_nested_dictionary(bkgd_jsdict, is_bkgd)
        #
        # dataset_bkgd = build_tfrecords_iterator(num_epochs, bkgd_train_path_pattern, is_bkgd, bkgd_feature,
        #                                         False, False, BKGD_SIZE,
        #                                         localization_bin_resolution, True)

        new_dataset = tf.data.Dataset.zip((dataset, new_dataset))

        # KEEP FOR BACKGROUND COMBINING
        # new_dataset = new_dataset.map(
        #     lambda x, y: combine_signal_and_noise_stacked_channel(x, y, 0, 48000, 8000, post_rectify=True))
        batch_sizes = tf.constant(16, dtype=tf.int64)
        new_dataset = new_dataset.shuffle(buffer_size=200).batch(batch_size=batch_sizes, drop_remainder=True)
        # combined_iter = new_dataset.make_one_shot_iterator()
        combined_iter = new_dataset.make_initializable_iterator()
        combined_iter_dict = combined_iter.get_next()
    ###END READING QUEUE MACHINERY###

    def make_downsample_filt_tensor(SR=16000, ENV_SR=200, WINDOW_SIZE=1001, beta=5.0, pycoch_downsamp=False):
        """
        Make the sinc filter that will be used to downsample the cochleagram
        Parameters
        ----------
        SR : int
            raw sampling rate of the audio signal
        ENV_SR : int
            end sampling rate of the envelopes
        WINDOW_SIZE : int
            the size of the downsampling window (should be large enough to go to zero on the edges).
        beta : float
            kaiser window shape parameter
        pycoch_downsamp : Boolean
            if true, uses a slightly different downsampling function
        Returns
        -------
        downsample_filt_tensor : tensorflow tensor, tf.float32
            a tensor of shape [0 WINDOW_SIZE 0 0] the sinc windows with a kaiser lowpass filter that is applied while downsampling the cochleagram
        """
        DOWNSAMPLE = SR / ENV_SR
        if not pycoch_downsamp:
            downsample_filter_times = np.arange(-WINDOW_SIZE / 2, int(WINDOW_SIZE / 2))
            downsample_filter_response_orig = np.sinc(downsample_filter_times / DOWNSAMPLE) / DOWNSAMPLE
            downsample_filter_window = signallib.windows.kaiser(WINDOW_SIZE, beta)
            downsample_filter_response = downsample_filter_window * downsample_filter_response_orig
        else:
            max_rate = DOWNSAMPLE
            f_c = 1. / max_rate  # cutoff of FIR filter (rel. to Nyquist)
            half_len = 10 * max_rate  # reasonable cutoff for our sinc-like function
            if max_rate != 1:
                downsample_filter_response = signallib.firwin(2 * half_len + 1, f_c, window=('kaiser', beta))
            else:  # just in case we aren't downsampling -- I think this should work?
                downsample_filter_response = zeros(2 * half_len + 1)
                downsample_filter_response[half_len + 1] = 1

            # Zero-pad our filter to put the output samples at the center  # n_pre_pad = int((DOWNSAMPLE - half_len % DOWNSAMPLE))  # n_post_pad = 0  # n_pre_remove = (half_len + n_pre_pad) // DOWNSAMPLE  # We should rarely need to do this given our filter lengths...  # while _output_len(len(h) + n_pre_pad + n_post_pad, x.shape[axis],  #                  up, down) < n_out + n_pre_remove:  #     n_post_pad += 1  # downsample_filter_response = np.concatenate((np.zeros(n_pre_pad), downsample_filter_response, np.zeros(n_post_pad)))

        downsample_filt_tensor = tf.constant(downsample_filter_response, tf.float32)
        downsample_filt_tensor = tf.expand_dims(downsample_filt_tensor, 0)
        downsample_filt_tensor = tf.expand_dims(downsample_filt_tensor, 2)
        downsample_filt_tensor = tf.expand_dims(downsample_filt_tensor, 3)

        return downsample_filt_tensor

    def downsample(signal, current_rate, new_rate, window_size, beta, post_rectify=True):
        downsample = current_rate / new_rate
        message = ("The current downsample rate {} is "
                   "not an integer. Only integer ratios "
                   "between current and new sampling rates "
                   "are supported".format(downsample))

        assert (current_rate % new_rate == 0), message
        message = ("New rate must be less than old rate for this "
                   "implementation to work!")
        assert (new_rate < current_rate), message
        # make the downsample tensor
        downsample_filter_tensor = make_downsample_filt_tensor(current_rate, new_rate, window_size,
                                                               pycoch_downsamp=False)
        downsampled_signal = tf.nn.conv2d(signal, downsample_filter_tensor, strides=[1, 1, downsample, 1],
                                          padding='SAME', name='conv2d_cochleagram_raw')
        if post_rectify:
            downsampled_signal = tf.nn.relu(downsampled_signal)

        return downsampled_signal

    def put_kernels_on_grid(kernel, pad=1):

        '''Visualize conv. filters as an image (mostly for the 1st layer).
        Arranges filters into a grid, with some paddings between adjacent filters.
        Args:
          kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
          pad:               number of black pixels around each filter (between them)
        Return:
          Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
        '''

        # get shape of the grid. NumKernels == grid_Y * grid_X
        def factorization(n):
            for i in range(int(sqrt(float(n))), 0, -1):
                if n % i == 0:
                    if i == 1: print('Who would enter a prime number of filters')
                    return (i, int(n / i))

        (grid_Y, grid_X) = factorization(kernel.get_shape()[3].value)
        print('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

        x_min = tf.reduce_min(kernel)
        x_max = tf.reduce_max(kernel)
        kernel = (kernel - x_min) / (x_max - x_min)

        # pad X and Y
        x = tf.pad(kernel, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]), mode='CONSTANT')

        # X and Y dimensions, w.r.t. padding
        Y = kernel.get_shape()[0] + 2 * pad
        X = kernel.get_shape()[1] + 2 * pad
        x = tf.pad(kernel, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]), mode='CONSTANT')

        # X and Y dimensions, w.r.t. padding
        Y = kernel.get_shape()[0] + 2 * pad
        X = kernel.get_shape()[1] + 2 * pad

        channels = kernel.get_shape()[2]

        # put NumKernels to the 1st dimension
        x = tf.transpose(x, (3, 0, 1, 2))
        # organize grid on Y axis
        x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

        # switch X and Y axes
        x = tf.transpose(x, (0, 2, 1, 3))
        # organize grid on X axis
        x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

        # back to normal order (not combining with the next step for clarity)
        x = tf.transpose(x, (2, 1, 3, 0))

        # to tf.image_summary order [batch_size, height, width, channels],
        #   where in this case batch_size == 1
        x = tf.transpose(x, (3, 0, 1, 2))

        # scaling to [0, 255] is not necessary for tensorboard
        return x

    # Many lines are commented out to allow for quick architecture changes
    # TODO:This should be abstracted to arcitectures are defined by some sort of
    # config dictionary or file

    def gradients_with_loss_scaling(loss, loss_scale):
        """Gradient calculation with loss scaling to improve numerical stability
        when training with float16.
        """

        grads = [(grad[0] / loss_scale, grad[1]) for grad in
                 tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-4).compute_gradients(loss * loss_scale,
                                                                                                     colocate_gradients_with_ops=True)]
        return grads

    # -> Copied foreground into combined_iter_dict[1] which should be the background, just to get correct shape here
    [L_channel, R_channel] = tf.unstack(combined_iter_dict[0]['train/image'], axis=3)
    concat_for_downsample = tf.concat([L_channel, R_channel], axis=0)
    reshaped_for_downsample = tf.expand_dims(concat_for_downsample, axis=3)

    # hard coding filter shape based on previous experimentation
    new_sig_downsampled = downsample(reshaped_for_downsample, sr, cochleagram_sr, window_size=4097, beta=10.06,
                                     post_rectify=post_rectify)
    downsampled_squeezed = tf.squeeze(new_sig_downsampled)
    [L_channel_downsampled, R_channel_downsampled] = tf.split(downsampled_squeezed, num_or_size_splits=2, axis=0)
    downsampled_reshaped = tf.stack([L_channel_downsampled, R_channel_downsampled], axis=3)
    new_sig_nonlin = tf.pow(downsampled_reshaped, 0.3)

    net = NetBuilder()
    out = net.build(config_array, new_sig_nonlin, training_state, dropout_training_state, filter_dtype, padding,
                    n_classes_localization, n_classes_recognition, False, None)

    combined_dict = collections.OrderedDict()
    combined_dict_fg = collections.OrderedDict()
    combined_dict_bkgd = collections.OrderedDict()
    for k, v in combined_iter_dict[0].items():
        if k != 'train/image' and k != 'train/image_height' and k != 'train/image_width':
            combined_dict_fg[k] = combined_iter_dict[0][k]
    for k, v in combined_iter_dict[1].items():
        if k != 'train/image' and k != 'train/image_height' and k != 'train/image_width':
            combined_dict_bkgd[k] = combined_iter_dict[1][k]
    combined_dict[0] = combined_dict_fg
    combined_dict[1] = combined_dict_bkgd  # never used!

    labels_batch_sphere = tf.add(tf.scalar_mul(tf.constant(36, dtype=tf.int32), combined_dict[0]['train/elev']),
                                 combined_dict[0]['train/azim'])
    labels_batch_cost_sphere = tf.squeeze(labels_batch_sphere)


    cost = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=labels_batch_cost_sphere))

    pred_dist = tf.nn.softmax(out)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        update_grads = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-4).minimize(cost)

    # Evaluate model
    pred_is_correct = tf.equal(tf.argmax(out, 1), tf.cast(labels_batch_cost_sphere, tf.int64))
    net_pred = tf.argmax(out, 1)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    config = tf.ConfigProto(allow_soft_placement=True, inter_op_parallelism_threads=0, intra_op_parallelism_threads=0)
    sess = tf.Session(config=config)
    sess.run(init_op)

    ##Testing loop
    ckpt_version = 100000
    sess.run(combined_iter.initializer)
    print("Starting model from checkpoint at", ckpt_version, "iterations.")
    batch_acc = []
    batch_conditional = []
    saver = tf.train.Saver(max_to_keep=None)
    saver.restore(sess, newpath + "/model.ckpt-" + str(ckpt_version))
    step = 0
    try:
        eval_vars = list(combined_dict[0].values())
        eval_keys = list(combined_dict[0].keys())
        while True:
            pic_evaluated, pd_evaluated, ev_evaluated, np_evaluated, lbcs_evaluated = sess.run([pred_is_correct, pred_dist, eval_vars, net_pred, labels_batch_cost_sphere])
            print('pic_evaluated: ', pic_evaluated)
            print('pd_evaluated: ', pd_evaluated)
            print('ev_evaluated: ', ev_evaluated)
            print('np_evaluated: ', np_evaluated)
            print('lbcs_evaluated: ', lbcs_evaluated, '\n')

            array_len = len(ev_evaluated)
            if isinstance(ev_evaluated, list):
                ev_evaluated = list(zip(*ev_evaluated))
                batch_conditional += [(cond, var) for cond, var in zip(pd_evaluated, ev_evaluated)]
                batch_acc += [(pd, ev) for pd, ev in zip(pic_evaluated, ev_evaluated)]
            else:
                ev_evaluated = np.array([np.squeeze(x) for x in ev_evaluated])
                split = np.vsplit(ev_evaluated, array_len)
                batch_conditional += [(cond, var) for cond, var in zip(pd_evaluated, ev_evaluated.T)]
                split.insert(0, pic_evaluated)
                batch_acc += np.dstack(split).tolist()[0]

            step += 1  # +1 for each batch
            if step % display_step == 0:  # display_step=25 -> every 25*16 = 400 steps
                print("Iter " + str(
                    step * batch_size))
            # Probably not going to reach here anyways for now
            if (step + 1) % write_step == 0:  # write_step=15625 -> every 15625*16 = 250K steps
                print("writing batch data at step: {}".format(step))
                write_batch_data(newpath, train_path_pattern, ckpt_version, batch_acc, batch_conditional, eval_keys,
                                 step)
                print("Data written")
                batch_acc = []
                batch_conditional = []
            if step == 500000:
                print("Break!")
                break
    except tf.errors.ResourceExhaustedError:
        print("Out of memory error")
        error = "Out of memory error"
        with open(newpath + '/test_error_{}.json'.format(ckpt_version), 'w') as f:
            json.dump(arch_ID, f)
            json.dump(error, f)
    except tf.errors.OutOfRangeError:
        print("Out of Range Error. Optimization Finished")

    finally:
        if train_path_pattern.split("/")[-2] == 'testset':
            stimuli_name = 'testset_' + train_path_pattern.split("/")[-3]
        else:
            stimuli_name = train_path_pattern.split("/")[-2]
        np.save(newpath + '/plot_array_padded_{}_iter{}.npy'.format(stimuli_name, ckpt_version), batch_acc)
        np.save(newpath + '/batch_conditional_{}_iter{}.npy'.format(stimuli_name, ckpt_version), batch_conditional)
        acc_corr = [pred[0] for pred in batch_acc]
        acc_accuracy = sum(acc_corr) / len(acc_corr)
        with open(newpath + '/accuracies_test_{}_iter{}.json'.format(stimuli_name, ckpt_version), 'w') as f:
            json.dump(acc_accuracy, f)
        with open(newpath + '/keys_test_{}_iter{}.json'.format(stimuli_name, ckpt_version), 'w') as f:
            json.dump(eval_keys, f)

    sess.close()
    tf.reset_default_graph()

if __name__ == "__main__":
    tf_record_CNN_spherical()