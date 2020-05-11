
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from util import *
from matplotlib import pyplot as plt
from PIL import Image
import time
import os
import sys
import threading
from ac import *

data_dir="D:\\png128\\"
work_path="D:\\workspace\\train model\\"
max_depth=3
USE_AUG=True
test_list_global=[[0,1,2,3,4,5,6,7,8],[0,1,2,3,4,5,6,7,8],[0,1,2,3,4,5,6,7,8]]
zscan_to_raster=[0,1,4,5,2,3,6,7,8,9,12,13,10,11,14,15]

ratio = 0.0005
size_list=[128,64,32]
num_model=[9,9,9]
img_test=[]
pdfs=[]
code_test = []
code_shape_test = []
code_noise_test = []

code_noise_shift=[]
rec_test = []
cdf=[]
code_bin=[]

for depth in range(max_depth):
    size = size_list[depth]
    img_test.append(tf.placeholder(tf.float32, shape=[1, size, size, 3], name="img_test_"+str(size)))
    code_bin.append(tf.placeholder(tf.float32, shape=[1, size>>4, size>>4, fea], name="code_test_"+str(size)))

    pdfs.append([])
    code_test.append([])
    code_shape_test.append([])
    code_noise_test.append([])
    code_noise_shift.append([])
    rec_test.append([])
    cdf.append([])


    for i in range(num_model[depth]):
        if i<num_model[depth]:
            code_test[depth].append(encoder(img_test[depth], 'subnet'+str(128)+'_' + str(i)))
            code_shape_test[depth].append(code_test[depth][i].shape.as_list())
            pdfs[depth].append(PDF_MULTI('subnet' + str(128) + '_' + str(i), minval=-15, maxval=15,category=code_shape_test[depth][i][-1]))
            code_noise_test[depth].append(pdfs[depth][i].shift_code(tf.reshape(tf.round(code_test[depth][i]),[-1,code_shape_test[depth][i][-2] * code_shape_test[depth][i][-3], code_shape_test[depth][i][-1]])))
            cdf[depth].append(pdfs[depth][i].get_cdf())
            rec_test[depth].append(tf.clip_by_value(tf.image.yuv_to_rgb(decoder(code_bin[depth], 'subnet'+str(128)+'_' + str(i)))*255,0,255))


init_op = tf.global_variables_initializer()


def rot(test_image,aug_idx):
    if (aug_idx<4):
        i_rot=aug_idx
        return np.rot90(test_image, k=i_rot, axes=(1, 2))
    else:
        i_rot=aug_idx-4
        return np.rot90(np.flip(test_image, axis=1), k=i_rot, axes=(1, 2))

def inv_rot(test_image,aug_idx):
    if (aug_idx<4):
        i_rot=aug_idx
        return np.rot90(test_image, k=4-i_rot, axes=(1, 2))
    else:
        i_rot=aug_idx-4
        return np.flip(np.rot90(test_image, k=4-i_rot, axes=(1, 2)), axis=1)

def get_patch(img,depth_idx,zscan_order):
    raster_idx=zscan_to_raster[zscan_order]
    x=raster_idx//4
    y=raster_idx%4
    size=size_list[depth_idx]
    x=x*32
    y=y*32
    return img[:,x:x+size,y:y+size,:]

def copy_to_pic(img,patch,depth_idx,zscan_order):
    raster_idx = zscan_to_raster[zscan_order]
    x = raster_idx // 4
    y = raster_idx % 4
    size = size_list[depth_idx]
    x=x*32
    y=y*32
    img[x:x + size, y:y + size, :]=patch

def init_coder():
    with tf.Session() as sess:
        module_file = tf.train.latest_checkpoint(work_path)
        if module_file is not None:
            sess.run(init_op)
            load_fn = tf.contrib.framework.assign_from_checkpoint_fn(module_file, var_list=tf.trainable_variables(), ignore_missing_vars=True)
            load_fn(sess)
        else:
            print("model not found")
            return



def xencode(test_image,image_shape,sess,f,m_index,aug_index,depth_index,pdf_index,b_full_ensemble,max_depth):
    fb = open(f +".bin", "wb")
    buffer_size = 1000000
    buffer = np.zeros([buffer_size], dtype=np.uint8)
    codec = Arithmetic_Codec(buffer_size, buffer)
    codec.start_encoder()
    codec.put_bits(image_shape[0], 16)
    codec.put_bits(image_shape[1], 16)
    image_shape[:2]=np.int32(np.ceil(image_shape[:2]/128)*128)

    if not b_full_ensemble:
        depth_index = np.tile(np.expand_dims(depth_index,0),[16,1])
        aug_index = np.tile(np.expand_dims(aug_index,0), [16, 1])
        m_index = np.tile(np.expand_dims(m_index,0), [16, 1])
        pdf_index = np.tile(np.expand_dims(pdf_index,0), [16, 1])

    for i in range(np.shape(test_image)[0]):
        img_org=np.expand_dims(test_image[i,:,:,:],0)
        depth_idx=depth_index[:,i]
        aug_idx = aug_index[:, i]
        m_idx = m_index[:, i]
        pdf_idx=pdf_index[:,i]
        j=0
        while(j < np.shape(depth_idx)[0]):
            img = get_patch(img_org, depth_idx[j], j)
            img = rot(img,aug_idx[j])
            code=sess.run(code_noise_test[depth_idx[j]][m_idx[j]],feed_dict={img_test[depth_idx[j]]: img})
            m_cdf=sess.run(cdf[depth_idx[j]][m_idx[j]])[pdf_idx[j],...]

            if b_full_ensemble:
                idx=(((depth_idx[j]*10+m_idx[j])<<10)+(aug_idx[j]<<7)+(pdf_idx[j]))
                codec.put_bits(idx, 15)
            bin = xencodeCU(codec, np.squeeze(code), np.squeeze(m_cdf))
            j=j+(np.shape(depth_idx)[0]>>(depth_idx[j]*2))

    code_len=codec.stop_encoder()
    fb.write(np.array(bin[:code_len],dtype=np.uint8).tobytes())
    fb.close()


def xdecode(sess,f,b_full_ensemble):
    fb = open(f + ".bin", "rb")
    buffer_size = 1000000
    buffer = np.zeros([buffer_size], dtype=np.uint8)
    bin = fb.read()
    buf = np.frombuffer(bin, dtype=np.uint8)
    code_len = buf.shape[0]
    buffer[:code_len] = buf.copy()
    codec = Arithmetic_Codec(buffer_size, buffer)
    codec.start_decoder()
    h=codec.get_bits(16)
    w=codec.get_bits(16)
    shape_pad=np.int32(np.ceil(np.array([h,w])/128)*128)
    out_img = np.zeros([shape_pad[0],shape_pad[1],3], dtype=np.uint8)
    width_in_ctu=np.ceil(w/128).astype(np.uint16)
    cu_idx=0
    while codec.ac_pointer<code_len:
        if b_full_ensemble:
            idx = codec.get_bits(15)
            pdf_idx=idx&127
            aug_idx=(idx>>7)&7
            m_idx=((idx>>10)&31)%10
            depth_idx=((idx>>10)&31)//10
        else:
            pdf_idx=0
            aug_idx=0
            m_idx=0
            depth_idx=0
        m_cdf = sess.run(cdf[depth_idx][m_idx])[pdf_idx,...]
        num_sym=(8>>depth_idx)**2
        code_dec=xdecodeCU(codec,num_sym,np.squeeze(m_cdf))
        code_dec=code_dec.astype(np.float32)
        rec = sess.run(rec_test[depth_idx][m_idx], feed_dict={code_bin[depth_idx]: np.reshape(np.transpose(code_dec - 26,[1,0]), [1, 8 >> depth_idx, 8 >> depth_idx, fea])})
        rec = np.round(inv_rot(rec, aug_idx))
        ras = cu_idx // 16
        zs = cu_idx % 16
        x = ras // width_in_ctu * 128 + zscan_to_raster[zs]// 4 * 32
        y = ras % width_in_ctu * 128 + zscan_to_raster[zs]% 4 * 32
        out_img[x:x + (128>>depth_idx), y:y + (128>>depth_idx), :] = rec[0, :, :, :]
        cu_idx += (16 >> (depth_idx * 2))

    codec.stop_decoder()
    plt.imsave(f + "_dec" + ".png", out_img[:h,:w,:])
    fb.close()

if __name__ == "__main__":
    init_coder()





