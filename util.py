import tensorflow as tf
from time import clock, sleep
import sys
from scipy import stats
import numpy as np
fea=128
from tensorflow.contrib.coder.python.ops import coder_ops
pmf_to_quantized_cdf = coder_ops.pmf_to_quantized_cdf
range_decode = coder_ops.range_decode
range_encode = coder_ops.range_encode
from tensorflow.python.ops import functional_ops
from tensorflow.python.framework import dtypes

def gen_normal(minval,maxval,height,width):
    temp=[]
    a=np.arange(minval,maxval+0.1,0.1)
    for i in range(height):
        temp.append(stats.norm.pdf(a[i],0,1))
    temp=np.array(temp)
    temp=np.tile(temp,(width,1))
    temp=temp.astype(np.float32)
    return temp.T

class PDF_MULTI(object):    #multi_probability
    def __init__(self, subnet_name, minval=-15, maxval=15, category=1):
        pdfs=np.load("D:\\workspace\\train model\\"+subnet_name + "pdf.npy").astype(np.float32)
        #pdfs = np.expand_dims(np.load(subnet_name + "pdf_test_mean.npy").astype(np.float32),0)
        self.minval = minval
        self.maxval = maxval
        self.subnet_name = subnet_name
        self.num_pdfs=np.shape(pdfs)[0]
        self.category = category
        self.n_param = self.maxval - self.minval + 1
        with tf.variable_scope(subnet_name+'multi_pdf', reuse=tf.AUTO_REUSE):
            self.param = tf.get_variable(
                name="pdfs",
                initializer=pdfs
            )

    def prob_i(self, input):
        with tf.name_scope("probability"):
            param_i=self.param
            param_i = param_i/ tf.tile(tf.expand_dims(tf.reduce_sum(param_i, axis=1),1),(1,self.n_param,1))
            idx = tf.round(input - self.minval + 1)
            pdf_idx = tf.expand_dims(tf.expand_dims(tf.range(self.num_pdfs, dtype=tf.int32), 1), 2) * tf.tile(
                tf.expand_dims(tf.ones_like(input, dtype=tf.int32), axis=0), (self.num_pdfs, 1, 1))
            idx = tf.tile(tf.expand_dims(tf.clip_by_value(idx, 0, self.maxval-self.minval+2),0),(self.num_pdfs,1,1))
            col1 = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(self.category, dtype=tf.int32), axis=0) * tf.ones_like(input, dtype=tf.int32),0),(self.num_pdfs,1,1))
            pdf_idx=tf.expand_dims(pdf_idx,axis=3)
            col = tf.expand_dims(col1, axis=3)
            idx_3d = tf.expand_dims(tf.cast(idx, tf.int32), axis=3)
            idx_3d = tf.concat([pdf_idx, idx_3d, col], axis=3)
            param_i = tf.concat([tf.zeros([self.num_pdfs, 1, self.category]), param_i, tf.zeros([self.num_pdfs, 1, self.category])],
                                axis=1)
            p = tf.gather_nd(param_i, idx_3d)
            return tf.maximum(p, 1e-9)

    def encode(self, input):
        with tf.name_scope("probability_encode"):
            range_offset=10
            param_i = self.param
            param_i = tf.concat([tf.ones([self.num_pdfs, range_offset, self.category])*1e-6, param_i+1e-6, tf.ones([self.num_pdfs, range_offset, self.category])*1e-6],axis=1)
            param_i = param_i / tf.tile(tf.expand_dims(tf.reduce_sum(param_i, axis=1), 1), (1, self.n_param+2*range_offset, 1))
            idx = tf.round(input - self.minval+range_offset + 1)
            idx = tf.cast(tf.clip_by_value(idx, 0, self.maxval - self.minval + 2), dtype=tf.int16)
            idx = tf.transpose(idx, [0, 2, 1])
            param_i = tf.concat([tf.zeros([self.num_pdfs, 1, self.category]), param_i, tf.zeros([self.num_pdfs, 1, self.category])], axis=1)
            pmf = tf.transpose(param_i, [0,2,1])
            cdf = tf.expand_dims(pmf_to_quantized_cdf(pmf, 16), 2)

            def xencode(data):
                def multi_cdf_encode(m_cdf):
                    return range_encode(data, m_cdf, 16)

                strings_multi_cdf = functional_ops.map_fn(
                    multi_cdf_encode, cdf, dtype=dtypes.string, back_prop=False)
                return strings_multi_cdf

            strings = functional_ops.map_fn(
                xencode, idx, dtype=dtypes.string, back_prop=False)
            return strings

    def decode(self, string, shape):
        with tf.name_scope("probability_deocde"):
            range_offset = 10
            param_i = self.param
            param_i = tf.concat([tf.ones([self.num_pdfs, range_offset, self.category]) * 1e-6, param_i+1e-6, tf.ones([self.num_pdfs, range_offset, self.category]) * 1e-6], axis=1)
            param_i = param_i / tf.tile(tf.expand_dims(tf.reduce_sum(param_i, axis=1), 1), (1, self.n_param+2*range_offset, 1))
            param_i = tf.concat([tf.zeros([self.num_pdfs, 1, self.category]), param_i, tf.zeros([self.num_pdfs, 1, self.category])], axis=1)
            pmf = tf.transpose(param_i, [0,2,1])
            cdf = tf.expand_dims(pmf_to_quantized_cdf(pmf, 16), 2)

            def xdecode(bins):
                def multi_cdf_decode(m_cdf):
                    return range_decode(bins, [shape[2],shape[1]], m_cdf, 16)

                codes_multi_cdf = functional_ops.map_fn(
                    multi_cdf_decode, cdf, dtype=dtypes.int16, back_prop=False)
                return codes_multi_cdf

            codes = functional_ops.map_fn(
                xdecode, string, dtype=dtypes.int16, back_prop=False)

            return codes+self.minval - 1-range_offset,cdf

    def get_cdf(self):
        with tf.name_scope("probability_cdf"):
            range_offset = 10
            param_i = self.param
            param_i = tf.concat([tf.ones([self.num_pdfs, range_offset, self.category]) * 1e-6, param_i+1e-6, tf.ones([self.num_pdfs, range_offset, self.category]) * 1e-6], axis=1)
            param_i = param_i / tf.tile(tf.expand_dims(tf.reduce_sum(param_i, axis=1), 1), (1, self.n_param+2*range_offset, 1))
            param_i = tf.concat([tf.zeros([self.num_pdfs, 1, self.category]), param_i, tf.zeros([self.num_pdfs, 1, self.category])], axis=1)
            pmf = tf.transpose(param_i, [0,2,1])
            cdf = tf.expand_dims(pmf_to_quantized_cdf(pmf, 16),2)

            return cdf

    def shift_code(self, input):
        with tf.name_scope("code_shift"):
            range_offset=10
            idx = tf.round(input - self.minval+range_offset + 1)
            idx = tf.cast(tf.clip_by_value(idx, 0, self.maxval - self.minval + 2 + 2 * range_offset), dtype=tf.int16)
            idx = tf.transpose(idx, [0, 2, 1])
            return idx


class PDF(object):#
    def __init__(self, subnet_name, minval=-18, maxval=18, category=1):
        self.minval = minval
        self.maxval = maxval
        self.subnet_name = subnet_name
        self.n_param = (self.maxval - self.minval) * 10 + 1
        self.category = category


        with tf.variable_scope(self.subnet_name,reuse=tf.AUTO_REUSE):
            self.param = tf.get_variable(
                name="spline_pre256",
                initializer=gen_normal(self.minval,self.maxval,self.n_param,self.category)
                #initializer=interpdf
            )

    def prob(self, input):#for training
        with tf.name_scope("probability_256"):
            idx = (input - self.minval) * 10 + 2
            down_idx = tf.floor(idx)
            down_idx = tf.clip_by_value(down_idx, 0, self.n_param + 3)
            up_idx = tf.clip_by_value(down_idx + 1, 0, self.n_param + 3)
            temp=tf.expand_dims(tf.range(self.category, dtype=tf.int32), axis=0)
            print(temp.get_shape().as_list())
            col1 = tf.expand_dims(tf.range(self.category, dtype=tf.int32), axis=0) * tf.ones_like(input, dtype=tf.int32)
            print(col1.get_shape().as_list())
            col = tf.expand_dims(col1, axis=2)
            down_idx_2d = tf.expand_dims(tf.cast(down_idx, tf.int32), axis=2)
            up_idx_2d = tf.expand_dims(tf.cast(up_idx, tf.int32), axis=2)
            down_idx_2d = tf.concat([down_idx_2d, col], axis=2)
            up_idx_2d = tf.concat([up_idx_2d, col], axis=2)

            param_pad = tf.concat([tf.zeros([2, self.category]), self.param, tf.zeros([2, self.category])], axis=0)
            lower = tf.gather_nd(param_pad, down_idx_2d)
            upper = tf.gather_nd(param_pad, up_idx_2d)

            return tf.maximum((idx - down_idx) * (upper - lower) + lower, 1e-10)

    def prob_i(self, input):# for test
        with tf.name_scope("probability_256"):
            param_i = self.param[0:self.n_param:10,:]
            mask=tf.cast(param_i>1e-10,dtype=tf.float32)
            param_i=tf.multiply(param_i,mask)
            param_i = param_i / tf.reduce_sum(param_i, axis=0)
            param_i = tf.concat([tf.zeros([1, self.category]), param_i, tf.zeros([1, self.category])], axis=0)
            idx = tf.round(input - self.minval+1)
            idx = tf.clip_by_value(idx, 0, self.maxval-self.minval+2)
            col1 = tf.expand_dims(tf.range(self.category, dtype=tf.int32), axis=0) * tf.ones_like(input, dtype=tf.int32)
            print(col1.get_shape().as_list())
            col = tf.expand_dims(col1, axis=2)
            idx_2d = tf.expand_dims(tf.cast(idx, tf.int32), axis=2)
            idx_2d = tf.concat([idx_2d, col], axis=2)
            p = tf.gather_nd(param_i, idx_2d)
            return tf.maximum(p, 1e-10)

    def encode(self, input):
        with tf.name_scope("probability_256"):
            param_i = self.param[0:self.n_param:10, :]
            param_i = param_i / tf.reduce_sum(param_i, axis=0)
            param_i = tf.concat([tf.zeros([1, self.category]), param_i, tf.zeros([1, self.category])],
                                axis=0)
            idx = tf.round(input - self.minval + 1)
            idx = tf.cast(tf.clip_by_value(idx, 0, self.maxval - self.minval + 2),dtype=tf.int16)
            idx=tf.transpose(idx,[0,2,1])
            pmf = tf.transpose(param_i, [1,0])
            cdf = tf.expand_dims(pmf_to_quantized_cdf(pmf, 16), 1)

            def xencode(data):
                return range_encode(data, cdf, 16)

            strings = functional_ops.map_fn(
                xencode, idx, dtype=dtypes.string, back_prop=False)
            return strings


    def norm(self):
        with tf.name_scope(self.subnet_name+"_Distribution_norm"):
            norm_op = tf.assign(self.param, self.param / tf.reduce_sum(self.param, axis=0, keep_dims=True) / 0.1)
            return norm_op


class GDNBase(object):#GDN
    def __init__(self, in_channels, out_channels):
        self.gamma = tf.get_variable(#gamma
            name="gamma",
            shape=[1, 1, in_channels, out_channels],
            dtype=tf.float32,
            initializer=tf.random_uniform_initializer()
        )
        self.beta = tf.get_variable(#beta
            name="beta",
            shape=[out_channels],
            dtype=tf.float32,
            initializer=tf.constant_initializer(value=2**-5)
        )

    def __call__(self, input):
        raise Exception("No calling the base function!")

    def norm(self):
        with tf.name_scope("GDN_norm"):
            op1 = tf.assign(self.beta, tf.maximum(self.beta, 2 ** -5))
            with tf.control_dependencies([tf.assign(self.gamma, tf.maximum(self.gamma, 2 ** -5))]):
                op2 = tf.assign(self.gamma, (self.gamma + tf.transpose(self.gamma, perm=[0, 1, 3, 2])) / 2.0)
            return tf.group(op1, op2)

class GDN(GDNBase):#GDN
    def __init__(self, in_channels, out_channels):
        GDNBase.__init__(self, in_channels, out_channels)

    def __call__(self, input):
        with tf.name_scope("GDN"):
            real_beta = self.beta**2 - 2**-10
            real_gamma = self.gamma**2 - 2**-10
            return input / tf.sqrt(tf.nn.bias_add(tf.nn.conv2d(input**2, real_gamma, [1, 1, 1, 1], "VALID"), real_beta))


class iGDN(GDNBase):#IGDN
    def __init__(self, in_channels, out_channels):
        GDNBase.__init__(self,  in_channels, out_channels)

    def __call__(self, input):
        with tf.name_scope("iGDN"):
            real_beta = self.beta**2 - 2**-10
            real_gamma = self.gamma**2 - 2**-10
            return input * tf.sqrt(tf.nn.bias_add(tf.nn.conv2d(input**2, real_gamma, [1, 1, 1, 1], "VALID"), real_beta))

class ConvBase(object):
    def __init__(self, height, width, first_channels, second_channels, factor):
        self.kernel = tf.get_variable(
            name="kernel",
            shape=[height, width, first_channels, second_channels],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer_conv2d()
        )
        self.factor = factor

    def __call__(self, input):
        raise Exception("No calling the base function!")

    def norm(self):
        with tf.name_scope("Conv_norm"):
            l2_norm = tf.sqrt(tf.reduce_sum(self.kernel**2, axis=(0, 1, 2), keep_dims=True))
            norm_op = tf.assign(self.kernel, self.kernel / l2_norm)
            return norm_op

class DownConv(ConvBase):
    def __init__(self, height, width, in_channels, out_channels, factor):
        ConvBase.__init__(self, height, width, in_channels, out_channels, factor)
        self.bias = tf.get_variable(
            name="bias",
            shape=[out_channels],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0, dtype=tf.float32)
        )

    def __call__(self, input):
        with tf.name_scope("conv_downsample"):
            return tf.nn.bias_add(tf.nn.conv2d(input, self.kernel, [1, self.factor, self.factor, 1], "SAME"), self.bias)

class UpConv(ConvBase):
    def __init__(self, height, width, in_channels, out_channels, factor):
        ConvBase.__init__(self, height, width, out_channels, in_channels, factor)
        self.bias = tf.get_variable(
            name="bias",
            shape=[out_channels],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0, dtype=tf.float32)
        )
        self.out_channels = out_channels

    def __call__(self, input):
        with tf.name_scope("conv_upsample"):
            # in_shape = input.get_shape().as_list()
            # out_shape = [in_shape[0], in_shape[1]*self.factor, in_shape[2]*self.factor, self.out_channels]
            out_shape=tf.stack([tf.shape(input)[0], tf.shape(input)[1]*self.factor, tf.shape(input)[2]*self.factor, self.out_channels])
            return tf.nn.bias_add(tf.nn.conv2d_transpose(input, self.kernel, out_shape, [1, self.factor, self.factor, 1], "SAME"), self.bias)

def encoder(input, subnet_name):
    with tf.variable_scope(subnet_name ,reuse=tf.AUTO_REUSE):

        with tf.variable_scope("Encoder_layer1"):
            conv1_op = DownConv(9, 9, 3, fea, 4)# we used 128 ﬁlters (size 9⇥9) in the ﬁrst stage, each subsampled by a factor of 4 vertically and horizontally
            gdn1_op = GDN(fea, fea)
            #gdn1_op = tf.contrib.layers.GDN(reparam_offset=2 ** -10)
            layer1 = gdn1_op(conv1_op(input))
            tf.add_to_collection("normalize_NET_256", conv1_op.norm())
            tf.add_to_collection("normalize_NET_256", gdn1_op.norm())


        with tf.variable_scope("Encoder_layer2"):#The remaining two stages retain the number of channels,but use ﬁlters operating across all input channels(5⇥5⇥128),with outputs subsampled by a factor of 2 in each dimension
            conv2_op = DownConv(5, 5, fea, fea, 2)
            gdn2_op = GDN(fea, fea)
            #gdn2_op = tf.contrib.layers.GDN(reparam_offset=2 ** -10)
            layer2 = gdn2_op(conv2_op(layer1))
            tf.add_to_collection("normalize_NET_256", conv2_op.norm())
            tf.add_to_collection("normalize_NET_256", gdn2_op.norm())


        with tf.variable_scope("Encoder_layer3"):
            conv3_op = DownConv(5, 5, fea, fea, 2)
            gdn3_op = GDN(fea, fea)
            #gdn3_op = tf.contrib.layers.GDN(reparam_offset=2 ** -10)
            layer3 = gdn3_op(conv3_op(layer2))
            tf.add_to_collection("normalize_NET_256", conv3_op.norm())
            tf.add_to_collection("normalize_NET_256", gdn3_op.norm())

        return layer3


def decoder(input,subnet_name):
    with tf.variable_scope(subnet_name,reuse=tf.AUTO_REUSE):

        with tf.variable_scope("Decoder_layer1"):
            igdn1_op = iGDN(fea, fea)
            #igdn1_op = tf.contrib.layers.GDN(inverse=True,reparam_offset=2 ** -10)
            conv1_op = UpConv(5, 5, fea, fea, 2)
            layer1 = conv1_op(igdn1_op(input))
            tf.add_to_collection("normalize_NET_256", igdn1_op.norm())
            tf.add_to_collection("normalize_NET_256", conv1_op.norm())

        with tf.variable_scope("Decoder_layer2"):
            igdn2_op = iGDN(fea, fea)
            #igdn2_op = tf.contrib.layers.GDN(inverse=True, reparam_offset=2 ** -10)
            conv2_op = UpConv(5, 5, fea, fea, 2)
            layer2 = conv2_op(igdn2_op(layer1))
            tf.add_to_collection("normalize_NET_256", igdn2_op.norm())
            tf.add_to_collection("normalize_NET_256", conv2_op.norm())

        with tf.variable_scope("Decoder_layer3"):
            igdn3_op = iGDN(fea, fea)
            #igdn3_op = tf.contrib.layers.GDN(inverse=True, reparam_offset=2 ** -10)
            conv3_op = UpConv(9, 9, fea, 3, 4)
            layer3 = conv3_op(igdn3_op(layer2))
            tf.add_to_collection("normalize_NET_256", igdn3_op.norm())
            tf.add_to_collection("normalize_NET_256", conv3_op.norm())

        return layer3


class Process_64(object):
    def __init__(self, words, n_iter=100):
        self.words = words
        self.n_iter = n_iter
        self.length = len(str(n_iter))

    def __enter__(self):
        self.start = clock()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        '''sys.stdout.write(" " * (len(self.words) + len(str(self.n_iter)) * 2 + 3) + "\r")
        print (self.words + ": {:>5.2f} s".format(clock() - self.start))'''

    def print_64(self, cur_iter=0):
        '''sys.stdout.write(" " * (len(self.words) + len(str(self.n_iter))*2 + 3) + "\r")
        sys.stdout.write(self.words + ": " + eval("'256{{:>{}d}}'.format(self.length)").format(cur_iter) + "/{}\r".format(self.n_iter))
        sys.stdout.flush()'''
