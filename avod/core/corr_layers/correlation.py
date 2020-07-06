import tensorflow as tf

_correlation_ops = tf.load_op_library(
    tf.resource_loader.get_path_to_datafile("../ops/build/correlation.so"))


def correlation(input_a, input_b, kernel_size=1, max_displacement=20, stride_1=1, stride_2=2, padding=20):
    '''
    :param input_a: (batch_size, height, width, channel)
    :param input_b: (batch_size, height, width, channel)
    :param kernel_size: default = 1
    :param max_displacement: refer to d in the article, D = 2d + 1, default = 20
    :param stride_1: stride between output_a and output_b, default = 1
    :param stride_2: dilation in region (2 * displacement + 1, 2 * displacement + 1), default = 2
    :param padding: default = 20
    :return: tensor(batch_size, out_height, out_width, out_channel)
                out_height = (in_height + 2 * padding - max_displacement * 2 - 1) / stride_1 + 1
                out_width = (in_width + 2 * padding - max_displacement * 2 - 1) / stride_1 + 1
                out_channel = (max_displacement / stride_2 * 2 + 1)**2
    '''
    return _correlation_ops.correlation(input_a,
                                        input_b,
                                        kernel_size,
                                        max_displacement,
                                        stride_1,
                                        stride_2,
                                        padding)


@tf.RegisterGradient("Correlation")
def _correlation_grad(corr_op, gradients):
    kernel_size = corr_op.get_attr("kernel_size")
    max_displacement = corr_op.get_attr("max_displacement")
    stride_1 = corr_op.get_attr("stride_1")
    stride_2 = corr_op.get_attr("stride_2")
    pad = corr_op.get_attr("pad")

    corr_grads = _correlation_ops.correlation_grad(gradients,
                                                   corr_op.inputs[0],
                                                   corr_op.inputs[1],
                                                   kernel_size,
                                                   max_displacement,
                                                   stride_1,
                                                   stride_2,
                                                   pad)

    # Return the gradients with respect to input_a and input_b
    return corr_grads.backprops_a, corr_grads.backprops_b
