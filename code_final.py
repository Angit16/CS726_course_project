
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Convolutional 3D layer definition
def conv3DLayer(input_layer, input_channels, output_channels, height, width, length, stride=1, activation=tf.nn.relu, padding="SAME", name="", is_training=True, M_pool = tf.nn.max_pool3d):
    with tf.variable_scope("conv3D" + name):
        kernel = tf.get_variable("weights", shape=[length, height, width, input_channels, output_channels], \
            dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable("bias", shape=[output_dim], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv3d(input_layer, kernel, stride, padding=padding)
  

        bias = tf.nn.bias_add(conv, b)
        # if activation:
        bias = activation(bias, name="activation")
        # Maxpool operation
        if M_pool:
        pooled = M_pool(bias, [1,2,2,2,1], strides=[1,2,2,2,1], name="M_pool")
        bias = batch_norm(pooled, is_training)

    return bias

# fully connected layer definition
def fully_connected(input_layer, shape, name="", is_training=True,dropout=tf.layers.dropout):
    with tf.variable_scope("fully" + name):
        kernel = tf.get_variable("weights", shape=shape, \
            dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
        fully = tf.matmul(input_layer, kernel)
        fully = tf.nn.relu(fully)
        if dropout:
        drop = dropout(inputs=fully, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
        fully = batch_norm(drop, is_training)
        return fully
   

class BNBLayer(object):
    def __init__(self):
        pass
# graph of the network
    def build_graph(self, voxel, activation=tf.nn.relu, is_training=True):
        # conv3D layer 1
        self.layer1 = conv3DLayer(voxel, 3, 96, 5, 5, 5, [1, 1, 1, 1, 1], name="layer1", activation=activation, is_training=is_training,M_pool=True)       
        # conv3D layer 2
        self.layer2 = conv3DLayer(self.layer1, 96, 192, 3, 3, 3, [1, 2, 2, 2, 1], name="layer2", activation=activation, is_training=is_training,M_pool=True)
        # conv3D layer 3
        self.layer3 = conv3DLayer(self.layer2, 192, 384, 3, 3, 3, [1, 2, 2, 2, 1], name="layer3", activation=activation, is_training=is_training,)
       # tensor reshaping

        self.flat= tf.reshape(self.layer3,[-1,24576])
        # fully connected layer 1
        self.fc1 = fully_connected(self.flat, 4096, name="fc1" ,is_training=is_training, dropout=True)
        # fully connected layer 2
        self.fc2 = fully_connected(self.fc1, 1024, name="fc2" ,is_training=is_training, dropout=True)
        # fully connected layer 3 or the output layer
        self.fc3 = fully_connected(self.fc2, 3*21, name="fc3" ,is_training=is_training, dropout=False)

# call the whole network in this definition
def ssd_model(sess, voxel_shape=(300, 300, 300),activation=tf.nn.relu, is_training=True):
    voxel = tf.placeholder(tf.float32, [None, voxel_shape[0], voxel_shape[1], voxel_shape[2], 1])
    phase_train = tf.placeholder(tf.bool, name='phase_train') if is_training else None
    with tf.variable_scope("3D_CNN_model") as scope:
        bnb_model = BNBLayer()
        bnb_model.build_graph(voxel, activation=activation, is_training=phase_train)

    if is_training:
        initialized_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="3D_CNN_model")
        sess.run(tf.variables_initializer(initialized_var))
    return bnb_model, voxel, phase_train

# squared L2_loss function
    def loss_func(model, joint_pose):
        # for i in range(400)
        transformed_model = tf.placeholder(tf.float32, model.fc3.get_shape().as_list()) 
        # g_map = tf.placeholder(tf.float32, model.cordinate.get_shape().as_list()[:4])
        difference = tf.subtract(model.fc3, joint_pose)
        total_loss_1 =  tf.nn.l2_loss(difference)
        total_loss_2 = total_loss_1 * 2

    return total_loss_2

# Optimization module
def create_optimizer(loss_func, lr=0.01):
    # Stochastic GradientDescentOptimizer with learning rate 0.01
    opt = tf.train.GradientDescentOptimizer(lr)  
    optimizer = opt.minimize(loss_func)
    return optimizer

# Training model definition
def train(batch_num, pcd_path = None, label_path=None,\
        dataformat="pcd", label_type="txt", scale=4, lr=0.01, \
# find out min_x, min_y, min_Z and min_x, min_y, min_Z respectively to input in the voxel_shape
        voxel_shape=(32, 32, 32), epoch=101): 
# input size is 32*32*32; x-y-z dimension limits are to be checked from the input data
    # tf Graph input
    batch_size = batch_num
    training_epochs = epoch

    with tf.Session() as sess:
        
        saver = tf.train.Saver()
        joint_pose = np.loadtxt(r'/CS726_course_project/transformed_joint.txt', delimiter =',')

        total_loss = loss_func4(model, joint_pose)
        optimizer = create_optimizer(total_loss, lr=lr)
        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in range(training_epochs):
                model, voxel, phase_train = ssd_model(sess, voxel_shape=voxel_shape, activation=tf.nn.relu, is_training=True)

                sess.run(optimizer)

                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(cc))
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(iol))
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(nol))
            if (epoch != 0) and (epoch % 10 == 0):
                print "Save epoch " + str(epoch)
                saver.save(sess, "velodyne_025_deconv_norm_valid" + str(epoch) + ".ckpt")
        print("Optimization Finished!")


# Test set up definition
def test(batch_num, label_path=None, calib_path=None, resolution=0.2, dataformat="pcd", label_type="txt", is_velo_cam=False, \
             scale=4, voxel_shape=(800, 800, 40)):
    batch_size = batch_num
    p = []
    pc = None
    bounding_boxes = None
    places = None
    rotates = None
    size = None
    proj_velo = None


    with tf.Session() as sess:
        is_training=None
        model, voxel, phase_train = ssd_model(sess, voxel_shape=voxel_shape, activation=tf.nn.relu, is_training=is_training)
        saver = tf.train.Saver()
        saver.restore(sess, last_model)

        objectness = model.objectness
        cordinate = model.cordinate
        y_pred = model.y
        objectness = sess.run(objectness, feed_dict={voxel: voxel_x})[0, :, :, :, 0]
        cordinate = sess.run(cordinate, feed_dict={voxel: voxel_x})[0]
        y_pred = sess.run(y_pred, feed_dict={voxel: voxel_x})[0, :, :, :, 0]
        print objectness.shape, objectness.max(), objectness.min()
        print y_pred.shape, y_pred.max(), y_pred.min()

        index = np.where(y_pred >= 0.995)
        print np.vstack((index[0], np.vstack((index[1], index[2])))).transpose()
        print np.vstack((index[0], np.vstack((index[1], index[2])))).transpose().shape

        centers = np.vstack((index[0], np.vstack((index[1], index[2])))).transpose()
        centers = sphere_to_center(centers, resolution=resolution, \
            scale=scale, min_value=np.array([x[0], y[0], z[0]]))
        corners = cordinate[index].reshape(-1, 8, 3) + centers[:, np.newaxis]
        print corners.shape
        print voxel.shape
        publish_pc2(pc, corners.reshape(-1, 3))



if __name__ == '__main__':
     pcd_path = "/home/Downloads/cvpr1_MSRAHandTrackingDB/Subject1/*.bin"  #image files
     label_path = "/CS726_course_project/transformed_joint.txt" # joint pose

# Traning
     train(400, pcd_path, label_path=label_path, dataformat="bin",\
            scale=8, voxel_shape=(32, 32, 32))

# Testing
    pcd_path = "/home/katou01/download/testing/velodyne/002397.bin"
    test(1, pcd_path, label_path=None, resolution=0.1, dataformat="bin", \
scale=8, voxel_shape=(32, 32, 32))
