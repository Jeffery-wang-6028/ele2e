#运行此程序时请勿运行Photoshop illustrator等需要占用GPU资源的程序
from scipy.io import loadmat
from util_separa_conv import *


import time
import os
import argparse
#os.environ['CUDA_VISIBLE_DEVICES']='2'


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--model_index", type=int, default=0,
    help="which model to train")
parser.add_argument(
    "--size", type=int, default=128,
    help="patch size")
parser.add_argument(
    "--lambda", type=float, default=0.0005,dest="lmbda",
    help="lambda")

args = parser.parse_args()
model_index=args.model_index
size=args.size
ratio=args.lmbda
data_dir="/gdata/wanngyf/png"+str(size)+"/"
work_path="/ghome/wanngyf/"
model_path="/ghome/wanngyf/model"+str(size)+"_"+str(ratio)+"/"+str(model_index)+"/"

def imread_raw(filename):
    img_f=tf.read_file(data_dir+filename)
    raw_img = tf.reshape(tf.image.decode_image(img_f),(size, size, 3))  # , channels=3)
    float_img = tf.cast(raw_img, tf.float32) / 255.0
    float_img = tf.image.rgb_to_yuv(float_img)
    return float_img

def get_filename(model_index):#生成输入图文件列表
    if model_index == 0:
        filename = list(np.load(work_path + "files" + str(size) + "_" + str(model_index) + ".npy"))
    else:
        filename = list(np.load(work_path + "files" + str(size) + "_" + str(model_index) + "_" + str(ratio) + ".npy"))
    return filename

def kodak():
    test=loadmat(data_dir+"kodak"+str(size)+"_3.mat")['batch']
    n_data=test.shape[0]
    #test = np.reshape(test,(n_data,256,256,3))
    test = tf.reshape(test, (n_data, size, size, 3))
    test_data = tf.cast(test, tf.float32) / 255.0
    #test_data = test.astype(np.float32) / 255.0
    with tf.Session() as sess:
        test_data = tf.image.rgb_to_yuv(test_data).eval(session=sess)#需要运行一个会话才可以把tensor转为数组
        test_data = test_data
    return test_data
    #return np.expand_dims(test_data[:,:,:,0],-1)

def get_trainable_var_list(subnet_name):
    var_list = tf.trainable_variables()
    var_list = [var for var in var_list if var.op.name.startswith(subnet_name)]
    return var_list


# Define the global variables

test_batch_size = 1
img_test = tf.placeholder(tf.float32, shape=[test_batch_size, size, size, 3], name="img_test")
# ratio=0.0005
test_data = kodak()#读入数据
# Define the optimizer


# 训练过程
#img_test = img #for deriving feature map
i=model_index
# 训练过程
img_list_no_batch=tf.reshape(tf.data.Dataset.from_tensor_slices(tf.constant(get_filename(i))).map(imread_raw).make_one_shot_iterator().get_next(), [1, size, size, 3])

code_test=encoder(img_test, "subnet"+str(size)+"_"+str(i))
code_shape_test=code_test.shape.as_list()
code_noise_test=tf.round(code_test) # 量化
rec_test=decoder(code_noise_test, "subnet"+str(size)+"_"+str(i))
pdfs=PDF("subnet"+str(size)+"_"+str(i),minval=-15, maxval=15, category=code_shape_test[-1])
prob_test=pdfs.prob_i(tf.reshape(code_noise_test, [-1, code_shape_test[-1]]))

rate_test=-tf.reduce_sum(tf.log(prob_test)/tf.log(2.0)) /size**2 /test_batch_size
distortion_test=tf.reduce_mean(tf.squared_difference(img_test, rec_test))
distortion_luma_test=tf.reduce_mean(tf.squared_difference(img_test[:,:,:,0], tf.clip_by_value(rec_test[:,:,:,0],0,1)))
distortion_chroma_test=tf.reduce_mean(tf.squared_difference(img_test[:, :, :, 1:], tf.clip_by_value(rec_test[:, :, :,1:],-0.5,0.5)))

init_op = tf.global_variables_initializer()  # 初始化变量操作
saver = tf.train.Saver(max_to_keep=2)

def calc_weight_RD():
    with tf.Session() as sess:
        module_file = tf.train.latest_checkpoint(model_path)
        if module_file is not None:  # 模型的路径存在就加载模型
            sess.run(init_op)  # 初始化，归一化网络参数
            # saver.restore(sess, module_file)
            load_fn = tf.contrib.framework.assign_from_checkpoint_fn(module_file, var_list=tf.trainable_variables(),
                                                                     ignore_missing_vars=True)  # 逐层增量训练使用了迁移学习的部分变量加载方法
            load_fn(sess)
            print("Restore the model from {}.".format(module_file))
        else:
            print("not found")
            return
        all_filenames=list(get_filename(model_index))
        length=np.shape(all_filenames)[0]
        RDlist=np.zeros(length)

        for i_test in range(length):
            img_buffer=sess.run(img_list_no_batch)
            r_, d_= sess.run([rate_test, distortion_test],feed_dict={img_test:img_buffer})
            RDlist[i_test]=d_ + ratio * r_
            if i_test % 1000 == 0:
                print("processing "+all_filenames[i_test])

        RD_sort_index=np.argsort(RDlist)
        for i in range(1,9):
            file=np.array(all_filenames)[RD_sort_index[int((i-1)*length/8):int(i*length/8)]]
            np.save(work_path+"files"+str(size)+"_"+str(i)+"_"+str(ratio)+".npy", file)

def output_code():
    with tf.Session() as sess:
        module_file = tf.train.latest_checkpoint(model_path)
        if module_file is not None:  # 模型的路径存在就加载模型
            sess.run(init_op)  # 初始化，归一化网络参数
            # saver.restore(sess, module_file)
            load_fn = tf.contrib.framework.assign_from_checkpoint_fn(module_file, var_list=tf.trainable_variables(),
                                                                     ignore_missing_vars=True)  # 逐层增量训练使用了迁移学习的部分变量加载方法
            load_fn(sess)
            print("Restore the model from {}.".format(module_file))
        else:
            print("未找到模型")
            return
        all_filenames=list(get_filename(model_index))
        length=np.shape(all_filenames)[0]
        feature_map = np.zeros(shape=(length, (size>>4),(size>>4), fea))
        print(length)
        for i_test in range(length):
            img_buffer=sess.run(img_list_no_batch)
            feature_map[i_test,:,:,:]=sess.run(code_noise_test,feed_dict={img_test:img_buffer})
            #print("processing "+all_filenames[i_test])
        np.save(work_path+"feature_map_"+str(model_index)+"_"+str(ratio)+".npy",feature_map)

if __name__ == "__main__":
    calc_weight_RD()




