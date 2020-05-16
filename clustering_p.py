import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
num_of_classes =128
keep_d=128
ratio=0.0005
def derive_pdf(model_index): #derive histogram from coefficients
    xfea=np.load("D:\\workspace\\train model\\feature_map_"+str(model_index)+"_"+str(ratio)+".npy")
    nfiles=np.shape(xfea)[0]
    size=np.shape(xfea)[1]
    nchs=np.shape(xfea)[3]
    h=np.zeros((nfiles,31*nchs))
    mask=np.ones((size,size,nchs))
    mask=tf.constant(mask*np.arange(nchs)*31,dtype=tf.float32)
    x = tf.placeholder(tf.float32, shape=[size, size, nchs], name="feature_map")
    hist=tf.histogram_fixed_width(x+mask,(-15.5,-15+31*nchs-0.5),31*nchs)
    with tf.Session() as sess:
        for i in range(nfiles):
            h[i,:]=sess.run(hist,feed_dict={x:xfea[i,:,:,:]})
    h/=(size*size)
    np.save("subnet128_"+str(model_index)+"_"+str(ratio)+"pdf_every.npy",np.transpose(np.reshape(h,(nfiles,-1,31)),[0,2,1]))

def sort_channel(model_index):
    hm= np.load("subnet128_"+str(model_index)+"pdf_mean.npy")
    hm[np.where(hm==0)]=1
    entropy=np.sum(hm*np.log(hm),axis=0)
    sort_idx=np.argsort(entropy)
    np.save("subnet128_"+str(model_index)+"channel_sort_index.npy",sort_idx)

def pca(x,dim):
    m= np.shape(x)[0]
    mean = np.mean(x,axis=0)
    x_new = x - mean
    cov = np.dot(x_new.T,x_new)/m
    s,u,v = tf.linalg.svd(tf.constant(cov,tf.float32))
    with tf.Session() as sess:
        s,v=sess.run([s,v])
        return np.dot(x,v[:,:dim])

def pre_processing_set(model_index,input,dim, b_usePCA):
    length = np.shape(input)[0]
    ht = np.reshape(input, (length, -1))
    if(b_usePCA):
        pca = PCA(n_components=dim)#use PCA to reduce dimension before clustering
        ht = pca.fit_transform(ht)
    return ht

def train_kmeans(model_index,input,num_of_classes):
    x =tf.placeholder(tf.float32, shape=np.shape(input),name="pdfs")
    model = KMeans(inputs=x, num_clusters=num_of_classes, initial_clusters="kmeans_plus_plus")#,distance_metric='cross_entropy_distance')
    training_graph = model.training_graph()

    if len(training_graph) > 6:
        (all_scores, cluster_idx, scores, cluster_centers_initialized,
         cluster_centers_var, init_op, train_op) = training_graph
    else:
        (all_scores, cluster_idx, scores, cluster_centers_initialized,
         init_op, train_op) = training_graph

    cluster_idx = cluster_idx[0]
    avg_distance = tf.reduce_mean(scores)
    init_vars = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_vars)
        sess.run(init_op,feed_dict={x:input})

        for i in range(500):
            _, d,label= sess.run([train_op, avg_distance, cluster_idx],feed_dict={x:input})
            if i % 10 == 0:
                print("step %i, avg distance: %f" % (i, d))
        return label

def clustering(model_index,label):
    h = np.load("subnet128_" + str(model_index)+"_"+str(ratio) + "pdf_every.npy")
    hcenter = np.zeros((num_of_classes,np.shape(h)[1],np.shape(h)[2]))
    for i in range(num_of_classes):
        hcenter[i,:,:]=np.mean(h[np.where(label==i)],0)
    np.save("subnet128_" + str(model_index) + "pdf.npy", hcenter)

if __name__=="__main__":
    for i in range(1,9):
        derive_pdf(i)
        ht = np.load("subnet128_" + str(i)+"_"+str(ratio)+"pdf_every.npy")
        ht = pre_processing_set(i, ht, 128, True)
        label=train_kmeans(i,ht,num_of_classes)
        clustering(i,label)


