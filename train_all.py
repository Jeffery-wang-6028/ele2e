#training
from scipy.io import loadmat
from util import *


model_index=0
size=128
data_dir="D:\\png"+str(size)+"\\"
work_path="D:\\workspace\\train model\\"
model_path=work_path

def imread(filename):
    img_f=tf.read_file(data_dir+filename)
    raw_img = tf.reshape(tf.image.decode_image(img_f),(size,size,3))
    raw_img = tf.image.random_flip_left_right(raw_img)
    raw_img = tf.image.random_flip_up_down(raw_img)
    float_img = tf.cast(raw_img, tf.float32) / 255.0
    float_img = tf.image.rgb_to_yuv(float_img)
    disturb_img = float_img + tf.random_uniform([size, size, 3], minval=-0.01, maxval=0.01)
    disturb_img = disturb_img
    return disturb_img

def imread_raw(filename):
    img_f=tf.read_file(data_dir+filename)
    raw_img = tf.reshape(tf.image.decode_image(img_f),(size,size,3)) # , channels=3)
    float_img = tf.cast(raw_img, tf.float32) / 255.0
    float_img = tf.image.rgb_to_yuv(float_img)
    return float_img

def get_filename(model_index):
    filename=list(np.load(work_path+"files128_0.npy"))
    return filename

def kodak():
    test=loadmat("D:\\png"+str(size)+"\\"+"kodak"+str(size)+"_3.mat")['batch']
    n_data=test.shape[0]

    test = tf.reshape(test, (n_data, size, size, 3))
    test_data = tf.cast(test, tf.float32) / 255.0

    with tf.Session() as sess:
        test_data = tf.image.rgb_to_yuv(test_data).eval(session=sess)
        test_data = test_data
    return test_data


def get_trainable_var_list(subnet_name):
    var_list = tf.trainable_variables()
    var_list = [var for var in var_list if var.op.name.startswith(subnet_name)]
    return var_list


# Define the global variables
batch_size = 8
test_batch_size = 1
img_test = tf.placeholder(tf.float32, shape=[test_batch_size, size, size, 3], name="img_test")
ratio = 0.0005
learning_rate=tf.Variable(float(1e-4),trainable=False,dtype=tf.float32)

test_data = kodak()
# Define the optimizer
net_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
pdf_optimizer = tf.train.GradientDescentOptimizer(1e-4)
learning_rate_decay_op1=learning_rate.assign(1e-5)
learning_rate_decay_op2=learning_rate.assign(1e-6)

i=model_index
filename = tf.constant(get_filename(i))
dataset = tf.data.Dataset.from_tensor_slices((filename)).map(imread)

img_list_no_batch=tf.reshape(tf.data.Dataset.from_tensor_slices(tf.constant(get_filename(i))).map(imread_raw).make_one_shot_iterator().get_next(), [1, size, size, 3])
img_and_list=dataset.repeat().shuffle(1000).batch(batch_size).make_one_shot_iterator().get_next()
img_list=tf.reshape(img_and_list,[batch_size, size, size, 3])


code=encoder(img_list, "subnet"+str(size)+"_"+str(i))
code_shape=code.shape.as_list()
code_noise=code + tf.random_uniform(code_shape, minval=-0.5, maxval=0.5)
rec=decoder(code_noise,"subnet"+str(size)+"_"+str(i))

pdfs=PDF("subnet"+str(size)+"_"+str(i),minval=-15, maxval=15, category=code_shape[-1])
prob = pdfs.prob(tf.reshape(code_noise, [-1, code_shape[-1]]))
# Define the loss
with tf.name_scope("distortion"+"subnet"+str(size)+"_"+str(i)):
    distortion=tf.reduce_mean(tf.squared_difference(img_list, rec))
with tf.name_scope("rate"+"subnet"+str(size)+"_"+str(i)):
    rate = -tf.reduce_sum(tf.log(prob) / tf.log(2.0)) / size ** 2 / batch_size
with tf.name_scope("pdf_estimate"+"subnet"+str(size)+"_"+str(i)):
    Approx = -tf.reduce_mean(tf.reduce_sum(tf.reshape(prob, [-1, code_shape[-1]]), axis=0))
# Define the operation
pretrain_op=net_optimizer.minimize(
    distortion, var_list=[var for var in get_trainable_var_list("subnet"+str(size)+"_"+str(i)) if var != pdfs.param]
)
train_net_op=net_optimizer.minimize(
    distortion+ratio*rate, var_list=[var for var in get_trainable_var_list("subnet"+str(size)+"_"+str(i)) if var != pdfs.param]#
)

norm_net_op=tf.get_collection("normalize_NET_256")
train_pdf_op=pdf_optimizer.minimize(Approx, var_list=pdfs.param)
norm_pdf_op = pdfs.norm()

code_test=encoder(img_test, "subnet"+str(size)+"_"+str(i))
code_shape_test=code_test.shape.as_list()
code_noise_test=tf.round(code_test)
rec_test=decoder(code_noise_test, "subnet"+str(size)+"_"+str(i))
prob_test = pdfs.prob_i(tf.reshape(code_noise_test, [-1, code_shape_test[-1]]))
rate_test = -tf.reduce_sum(tf.log(prob_test) / tf.log(2.0)) / size ** 2 / test_batch_size

distortion_test=tf.reduce_mean(tf.squared_difference(img_test, rec_test))
distortion_luma_test=tf.reduce_mean(tf.squared_difference(img_test[:,:,:,0], tf.clip_by_value(rec_test[:,:,:,0],0,1)))
distortion_chroma_test=tf.reduce_mean(tf.squared_difference(img_test[:, :, :, 1:], tf.clip_by_value(rec_test[:, :, :,1:],-0.5,0.5)))
# Define the Saver
init_op = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=2)

def train():
    global_step=0
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        module_file = tf.train.latest_checkpoint(model_path)
        if module_file is not None:
            sess.run(init_op)
            load_fn = tf.contrib.framework.assign_from_checkpoint_fn(module_file, var_list=tf.trainable_variables(), ignore_missing_vars=True)
            load_fn(sess)
            print("Restore the model from {}.".format(module_file))
            global_step = int(module_file.split("-")[-1])
            if (global_step >= 10000):
                sess.run(learning_rate_decay_op1)
            if (global_step >= 15000):
                sess.run(learning_rate_decay_op2)
        else:
            sess.run(init_op)
            sess.run(norm_pdf_op)
            sess.run(norm_net_op)

        for i in range(2500):
            for _ in range(100):
                sess.run(pretrain_op)
                sess.run(norm_net_op)
            print("pretrain "+str(i))
        for _ in range(1000):
            sess.run(train_pdf_op)
            sess.run(norm_pdf_op)

        for epoch in range(global_step,17500):
            if (epoch == 10000):
                sess.run(learning_rate_decay_op1)
            if (epoch == 15000):
                sess.run(learning_rate_decay_op2)

            for j in range(100):
                with Process_64("net", 1) as p:
                    for i in range(p.n_iter):
                        #p.print_64(i)
                        sess.run(train_net_op)
                        sess.run(norm_net_op)
                with Process_64("pdf", 1) as p:
                        #p.print_64(i)
                        sess.run(train_pdf_op)
                        sess.run(norm_pdf_op)
            if epoch % 10 == 0:
                print("Epoch_deep {}:".format(epoch))
                r_list_picture = []
                d_list_picture = []
                d_luma_list_picture = []
                psnr_luma_list_picture = []
                for i_test in range(0, test_data.shape[0], 24):
                    r_list_block = []
                    d_list_block = []
                    d_luma_list_block = []
                    for j_test in range(24):
                        r_, d_, d_luma = sess.run(
                            [rate_test, distortion_test, distortion_luma_test],
                            {img_test: np.expand_dims(test_data[i_test + j_test, :, :, :], 0)})

                        r_list_block.append(r_)
                        d_list_block.append(d_)
                        d_luma_list_block.append(d_luma)

                    r_list_picture.append(np.mean(r_list_block))
                    d_list_picture.append(np.mean(d_list_block))
                    d_luma_list_picture.append(np.mean(d_luma_list_block))
                    psnr_luma_list_picture.append(10 * np.log10(1 / np.mean(d_luma_list_block)))

                r_mean = np.mean(r_list_picture)
                d_mean = np.mean(d_list_picture)
                d_mean_luma = np.mean(d_luma_list_picture)
                psnr_luma = np.mean(psnr_luma_list_picture)

                print("subnet"+str(model_index)+" 05_psnr64_LUMA: {:>5.2f} dB  bpp: {:>4.2f}  MSE_luma: {:>6.4f}  Total: {:>6.4f}"
                      .format(psnr_luma, r_mean, d_mean_luma, d_mean + ratio * r_mean))
                f = open(work_path+'record_r.txt', 'a+')
                f.write('Iter ')
                f.write(str(epoch) + ' ')
                f.write('PSNR ')
                f.write(str(psnr_luma) + ' ')
                f.write('bpp ')
                f.write(str(r_mean)+ ' ')
                f.write('MSE_luma ')
                f.write(str(d_mean_luma) + ' ')
                f.write('RD-cost ')
                f.write(str(d_mean + ratio * r_mean))
                f.write('\n')
                f.close()
                saver.save(sess, model_path+"model"+str(size)+"_"+str(model_index)+".ckpt", global_step=epoch)
        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    train()





