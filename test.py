#testing
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from util import *
from matplotlib import pyplot as plt
from PIL import Image
from encode import *
import time
import os
import sys
import threading
ratio = 0.0005
t_all=0
b_full_ensemble=True
data_dir="D:\\png128\\"
work_path="D:\\workspace\\train model\\"
b_range_coding=False #default False
b_output_index=True

if b_full_ensemble:#full_ensemble
    max_depth=3
    b_MULTI_PDF=True
    USE_AUG=True
    test_list_global=[[0,1,2,3,4,5,6,7,8],[0,1,2,3,4,5,6,7,8],[0,1,2,3,4,5,6,7,8]]
else:#anchor
    max_depth = 1
    b_MULTI_PDF = False
    USE_AUG = False
    test_list_global = [[0], [0], [0]]

file_list=os.listdir(work_path+"test_image")

def divide_image(im_input):  #divide image into 128x128 blocks
    h=np.int32(im_input.shape[0]/128)
    w=np.int32(im_input.shape[1]/128)
    batch=h*w
    img_batched=np.zeros([batch,128,128,3])
    for i in range(h):
        for j in range(w):
            img_batched[i*w+j,...]=im_input[i*128:(i+1)*128,j*128:(j+1)*128,...]
    return  img_batched

def import_image(filename):
    img_f = tf.read_file(work_path +"test_image\\"+ filename)
    raw_img = tf.image.decode_image(img_f)
    s=tf.to_float(tf.shape(raw_img))
    im=tf.image.pad_to_bounding_box(raw_img, 0,0,tf.to_int32(tf.ceil(s[0] / 128))*128, tf.to_int32(tf.ceil(s[1] / 128))*128)
    im=tf.image.rgb_to_yuv(tf.cast(im, tf.float32) / 255.0)
    return im,s
# Define the global variables

filename = tf.constant(file_list)
test_dataset = tf.data.Dataset.from_tensor_slices((filename)).map(import_image) #reading test pngs
img_org=test_dataset.make_one_shot_iterator().get_next()
size_list=[128,64,32]
test_batch_size_list=[None,None,None]
num_model=[9,9,9]
img_test=[]
pdfs=[]
code_test = []
code_shape_test = []
code_noise_test = []
rec_test = []
rate_test=[]
distortion_test=[]
distortion_luma_test = []
distortion_chroma_test = []
prob_test=[]

for depth in range(max_depth):
    size = size_list[depth]
    img_test.append(tf.placeholder(tf.float32, shape=[None, size, size, 3], name="img_test_"+str(size)))
    pdfs.append([])
    code_test.append([])
    code_shape_test.append([])
    code_noise_test.append([])
    rec_test.append([])
    rate_test.append([])
    distortion_test.append([])
    distortion_luma_test.append([])
    distortion_chroma_test.append([])
    prob_test.append([])

    for i in range(num_model[depth]):
        if i<num_model[depth]:
            code_test[depth].append(encoder(img_test[depth], 'subnet'+str(128)+'_' + str(i)))
            code_shape_test[depth].append(code_test[depth][i].shape.as_list())
            code_noise_test[depth].append(tf.round(code_test[depth][i]))  # quantization
            rec_test[depth].append(decoder(code_noise_test[depth][i], 'subnet'+str(128)+'_' + str(i)))

            if(b_MULTI_PDF):
                pdfs[depth].append(PDF_MULTI('subnet'+str(128)+'_' + str(i), minval=-15, maxval=15, category=code_shape_test[depth][i][-1]))
            else:
                pdfs[depth].append(PDF('subnet' + str(128) + '_' + str(i), minval=-15, maxval=15, category=code_shape_test[depth][i][-1]))
            prob_test[depth].append(pdfs[depth][i].prob_i(tf.reshape(code_noise_test[depth][i], [-1, code_shape_test[depth][i][-1]])))

            if (b_MULTI_PDF):
                code_shape_test[depth][i].insert(0,-1)
                prob_test[depth][i] = tf.reshape(prob_test[depth][i], [128,-1,code_shape_test[depth][i][-3],code_shape_test[depth][i][-2],code_shape_test[depth][i][-1]])
            else:
                prob_test[depth][i] = tf.reshape(prob_test[depth][i],
                                                 [-1, code_shape_test[depth][i][-3], code_shape_test[depth][i][-2],
                                                  code_shape_test[depth][i][-1]])
            if (b_MULTI_PDF):
                if (b_range_coding):
                    rate_test[depth].append(pdfs[depth][i].encode(tf.reshape(code_noise_test[depth][i], [-1, code_shape_test[depth][i][-2]*code_shape_test[depth][i][-3],code_shape_test[depth][i][-1]])))
                else:
                    rate_test[depth].append(-tf.reduce_sum(tf.log(prob_test[depth][i])/tf.log(2.0),[2,3]) /size**2)
            else:
                if (b_range_coding):
                    rate_test[depth].append(pdfs[depth][i].encode(tf.reshape(code_noise_test[depth][i], [-1, code_shape_test[depth][i][-2]*code_shape_test[depth][i][-3],code_shape_test[depth][i][-1]])))
                else:
                    rate_test[depth].append(-tf.reduce_sum(tf.log(prob_test[depth][i]) / tf.log(2.0), [1, 2, 3]) / size ** 2)
            distortion_test[depth].append(tf.reduce_mean(tf.squared_difference(img_test[depth], rec_test[depth][i]),[1,2,3]))
            distortion_luma_test[depth].append(tf.reduce_mean(tf.squared_difference(img_test[depth][:,:,:,0], rec_test[depth][i][:,:,:,0]),[1,2]))
            distortion_chroma_test[depth].append(tf.reduce_mean(tf.squared_difference(img_test[depth][:, :, :, 1:], rec_test[depth][i][:, :, :,1:]),[1,2,3]))
    # Define the Saver

init_op = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=1)


def length(input):
    return list(map(lambda x: len(x), input))

def reorder_indexmap(index_map,out_batchsize):
    if(len(np.shape(index_map))<2):
        index_map=np.expand_dims(index_map,0)
    return np.concatenate((index_map[:,:out_batchsize],index_map[:,out_batchsize:2*out_batchsize],index_map[:,2*out_batchsize:3*out_batchsize],index_map[:,3*out_batchsize:]),axis=0)

def split_image_into_sub_block(test_image):
    image_shape=np.shape(test_image)
    size_sub=image_shape[1]>>1
    sub_image=np.concatenate((test_image[:,:size_sub,:size_sub,:],test_image[:,:size_sub,size_sub:,:],test_image[:,size_sub:,:size_sub,:],test_image[:,size_sub:,size_sub:,:]),axis=0)#Z字扫描
    return sub_image

def RDO(test_image,sess,test_list,depth):
    test_batch_size = test_batch_size_list[depth]
    r_list = []
    d_list = []
    rd_list = []
    d_luma_list = []
    d_chroma_list = []
    aug_index=np.array(np.ones(test_batch_size)*0,dtype=np.int32)#geometric self-ensemble idx
    pdf_idx_list = []
    depth_index=np.array(np.ones(test_batch_size)*depth,dtype=np.int32) #block depth idx
    for i_subnet in test_list:
        if not USE_AUG:
            r_, d_, d_luma, d_chroma,pdf_idx = check_RD(test_image, i_subnet, depth, sess)
        else:
            r_, d_, d_luma, d_chroma, aug_index,pdf_idx = check_RD_augmentation(test_image, i_subnet,depth,sess)
        r_list.append(r_)
        d_list.append(d_)
        d_luma_list.append(d_luma)
        d_chroma_list.append(d_chroma)
        rd_list.append(d_ + ratio * r_)
        pdf_idx_list.append(pdf_idx)
    opt_idx = np.array(np.where(rd_list == np.min(rd_list, axis=0)))
    opt_idx = opt_idx.T[np.lexsort(opt_idx)].T
    g = np.append(1, np.diff(opt_idx[1, :], axis=0))
    opt_idx = tuple(np.squeeze(opt_idx[:, np.where(g)]))
    m_index = np.array(opt_idx[0])                                      #network model idx
    pdf_index = np.array(pdf_idx_list)[opt_idx]                         #probability distribution model idx
    r_best = np.array(r_list)[opt_idx] + (np.log2(len(test_list))+1) / size_list[depth] ** 2
    d_best = np.array(d_list)[opt_idx]
    d_luma_best = np.array(d_luma_list)[opt_idx]
    d_chroma_best = np.array(d_chroma_list)[opt_idx]
    rd_best = d_best + ratio * r_best
    if (depth<max_depth-1):
        depth+=1
        test_image_split = split_image_into_sub_block(test_image)
        r_sub,d_sub,d_luma_sub,d_chroma_sub,m_index_sub,aug_index_sub,depth_index_sub,pdf_index_sub=RDO(test_image_split, sess, test_list_global[depth], depth)

        if b_output_index:
            m_index_sub=reorder_indexmap(m_index_sub,test_batch_size)
            aug_index_sub = reorder_indexmap(aug_index_sub,test_batch_size)
            depth_index_sub =reorder_indexmap(depth_index_sub,test_batch_size)
            pdf_index_sub = reorder_indexmap(pdf_index_sub, test_batch_size)

        r_sub = np.mean(np.reshape(r_sub,[-1,test_batch_size]),axis=0)
        d_sub = np.mean(np.reshape(d_sub, [-1,test_batch_size]), axis=0)
        d_luma_sub = np.mean(np.reshape(d_luma_sub, [-1,test_batch_size]), axis=0)
        d_chroma_sub=np.mean(np.reshape(d_chroma_sub,[-1,test_batch_size]),axis=0)
        rd_sub = d_sub + ratio * r_sub
        r_best, d_best, d_luma_best, d_chroma_best = np.where(rd_sub < rd_best,
                                                          [r_sub, d_sub, d_luma_sub, d_chroma_sub],
                                                          [r_best, d_best, d_luma_best, d_chroma_best])
        if b_output_index:
            m_index = np.where(rd_sub < rd_best,m_index_sub, m_index)
            aug_index = np.where(rd_sub < rd_best, aug_index_sub, aug_index)
            depth_index = np.where(rd_sub < rd_best, depth_index_sub, depth_index)
            pdf_index = np.where(rd_sub < rd_best, pdf_index_sub, pdf_index)
    return r_best, d_best, d_luma_best, d_chroma_best,m_index, aug_index, depth_index, pdf_index

def check_RD(test_image,i_subnet,depth,sess):
    global t_all
    feed_dict={img_test[depth]: test_image}
    output_list = [rate_test[depth][i_subnet], distortion_test[depth][i_subnet], distortion_luma_test[depth][i_subnet], distortion_chroma_test[depth][i_subnet]]
    r, d, dluma, dchroma = sess.run(output_list,feed_dict)
    if not b_MULTI_PDF:
        if(b_range_coding):#default False
            r = np.array(list(map(lambda x: len(x), r))) * 8 / size_list[depth] ** 2
        pdf_idx = np.zeros(test_batch_size_list[depth],dtype=np.int32)
    else:                                                                       #numlti-probability distribution model
        if (b_range_coding):#default False
            r = np.array(list(map(lambda x: length(x), r))) * 8 / size_list[depth] ** 2
            pdf_idx = np.array(np.where(r == np.min(r, axis=1)))
            pdf_idx = pdf_idx.T[np.lexsort(pdf_idx)].T
            g = np.append(1, np.diff(pdf_idx[1, :], axis=0))
            pdf_idx = tuple(np.squeeze(pdf_idx[:, np.where(g)]))[0]
            r = np.min(r, 1) + 7 / size_list[depth] ** 2
        else:
            r=np.sum(r, -1)
            pdf_idx = np.array(np.where(r == np.min(r, axis=0)))
            pdf_idx = pdf_idx.T[np.lexsort(pdf_idx)].T
            g = np.append(1, np.diff(pdf_idx[1, :], axis=0))
            pdf_idx = tuple(np.squeeze(pdf_idx[:, np.where(g)]))[0]
            r = np.min(r, 0)+ 7 / size_list[depth] ** 2
    return r, d, dluma, dchroma, pdf_idx

def check_RD_augmentation(test_image,i_subnet,depth,sess):
    r_list = []
    d_list = []
    rd_list = []
    d_luma_list = []
    d_chroma_list = []
    pdf_idx_list=[]
    for i_flip in range(2):  # self_ensemble
        for i_rot in range(4):
            if (i_flip == 0):
                r_, d_, d_luma, d_chroma,pdf_idx = check_RD(np.rot90(test_image, k=i_rot, axes=(1, 2)),i_subnet, depth, sess)
            else:
                r_, d_, d_luma, d_chroma,pdf_idx = check_RD(np.rot90(np.flip(test_image, axis=1),k=i_rot, axes=(1, 2)), i_subnet, depth, sess)
            pdf_idx_list.append(pdf_idx)
            r_list.append(r_)
            d_list.append(d_)
            d_luma_list.append(d_luma)
            d_chroma_list.append(d_chroma)
            rd_list.append(d_ + ratio * r_)
    opt_idx = np.array(np.where(rd_list == np.min(rd_list, axis=0)))
    opt_idx = opt_idx.T[np.lexsort(opt_idx)].T
    g = np.append(1, np.diff(opt_idx[1, :], axis=0))
    opt_idx = tuple(np.squeeze(opt_idx[:, np.where(g)]))
    aug_index = np.array(opt_idx[0])
    return np.array(r_list)[opt_idx]+ 3 / size_list[depth] ** 2,np.array(d_list)[opt_idx],np.array(d_luma_list)[opt_idx],np.array(d_chroma_list)[opt_idx],aug_index,np.array(pdf_idx_list)[opt_idx]

def test():
    with tf.Session() as sess:
        module_file = tf.train.latest_checkpoint(work_path)
        if module_file is not None:  # load model
            sess.run(init_op)
            load_fn = tf.contrib.framework.assign_from_checkpoint_fn(module_file, var_list=tf.trainable_variables(), ignore_missing_vars=True)
            load_fn(sess)
            print("Restore the model from {}.".format(module_file))
        else:
            print("model not found")
            return
        i = 0
        global test_batch_size_list
        start=time.clock()
        r_list_picture = []
        d_list_picture = []
        d_luma_list_picture = []
        psnr_luma_list_picture=[]
        d_chroma_list_picture = []
        psnr_chroma_list_picture = []
        psnr_yuv_list_picture = []

        for file in file_list:  # image list
            file=file[:-4]
            test_data,image_shape = sess.run(img_org)
            test_data=divide_image(test_data)
            test_batch_size_list = [test_data.shape[0], test_data.shape[0]<<2, test_data.shape[0]<<4]
            r_best, d_best, d_luma_best, d_chroma_best,m_index,aug_index,depth_index,pdf_index= RDO(test_data,sess,test_list_global[0],0)
            # xencode(test_data,image_shape.astype(np.int32),sess,work_path +"test_image\\"+file,m_index,aug_index,depth_index,pdf_index,b_full_ensemble,max_depth) #for writing codebins
            # xdecode(sess,work_path +"test_image\\"+file,b_full_ensemble) #for decoding from codebins

            i+=1
            r_list_picture.append(np.mean(r_best))
            d_list_picture.append(np.mean(d_best))
            d_luma_list_picture.append(np.mean(d_luma_best))
            d_chroma_list_picture.append(np.mean(d_chroma_best))
            psnr_luma_list_picture.append(10 * np.log10(1 / np.mean(d_luma_best)))
            psnr_chroma_list_picture.append(10 * np.log10(1 / np.mean(d_chroma_best)))
            psnr_yuv_list_picture.append(10 * np.log10(1 / np.mean(d_best)))

        r_mean = np.mean(r_list_picture)
        d_mean = np.mean(d_list_picture)
        d_luma_mean = np.mean(d_luma_list_picture)
        d_chroma_mean = np.mean(d_chroma_list_picture)
        psnr_luma_mean = np.mean(psnr_luma_list_picture)
        psnr_chroma_mean = np.mean(psnr_chroma_list_picture)
        psnr_yuv_mean = np.mean(psnr_yuv_list_picture)
        t=time.clock()-start
        print("05_psnr64_LUMA: {} dB  psnr64_CHROMA: {} dB psnr bpp: {}  MSE_luma: {}  MSE_chroma: {} psnr_yuv: {} Total: {}  Time: {}"
            .format(psnr_luma_mean, psnr_chroma_mean, r_mean, d_luma_mean, d_chroma_mean, psnr_yuv_mean, d_mean + ratio * r_mean, t))



if __name__ == "__main__":
    init_coder()
    test()






