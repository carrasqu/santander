import tensorflow as tf
import input_data
import sys

#"images"
featurel=370

#parameters of the network
numberlabels=2
hiddenunits1=100
lamb=0.0005 # regularization parameter

# how does the data look like
#Ntemp=40 # number of different temperatures used in the simulation
#samples_per_T=250 # number of samples per temperature value
#Nord=20 # number of temperatures in the ordered phase

#reading the data in the directory txt 
mnist = input_data.read_data_sets(numberlabels,featurel,'txt', one_hot=True)

print "reading sets ok"

#sys.exit("pare aqui")

# defining weighs and initlizatinon
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.2)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.2, shape=shape)
  return tf.Variable(initial)

# defining the layers
def layers(x, W,b):
  return  tf.nn.relu(tf.matmul(x, W)+b)
         

# defining the model

x = tf.placeholder("float", shape=[None, featurel])
y_ = tf.placeholder("float", shape=[None, numberlabels])

#first layer 
#weights and bias
W_1 = weight_variable([featurel,hiddenunits1])
b_1 = bias_variable([hiddenunits1])

#Apply a sigmoid

O1 = layers(x, W_1,b_1)

#second layer(output layer in this case)
W_2 = weight_variable([hiddenunits1,numberlabels])
b_2 = bias_variable([numberlabels])

y_conv=tf.nn.softmax(tf.matmul(O1, W_2)+b_2 )


#Train and Evaluate the Model

# cost function to minimize (with L2 regularization)
bsize=50
#cross_entropy = tf.reduce_mean(-y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0) )-(1.0-y_)*tf.log(tf.clip_by_value(1.0-y_conv,1e-10,1.0))) +(lamb)*(tf.nn.l2_loss(W_1)+tf.nn.l2_loss(W_2))

cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))+(lamb)*(tf.nn.l2_loss(W_1)+tf.nn.l2_loss(W_2)) 


#defining the optimizer
#class tf.train.GradientDescentOptimizer
#class tf.train.AdagradOptimizer
#class tf.train.MomentumOptimizer
#class tf.train.AdamOptimizer
#class tf.train.FtrlOptimizer
#class tf.train.RMSPropOptimizer
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#optimizer = tf.train.GradientDescentOptimizer(0.5)
optimizer= tf.train.AdamOptimizer(0.0005)

train_step = optimizer.minimize(cross_entropy)

#predictions
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()
sess.run(tf.initialize_all_variables())


# training
for i in range(35000):

  batch = mnist.train.next_batch(bsize)
  #batch=(mnist.train.images[:,:].reshape(bsize,lx*lx), mnist.train.labels[:,:].reshape((bsize,numberlabels)) )
  if i%100 == 0:
    train_accuracy = sess.run(accuracy,feed_dict={
        x:batch[0], y_: batch[1]})
    print "step %d, training accuracy %g"%(i, train_accuracy)
    print sess.run(cross_entropy,feed_dict={x: batch[0], y_: batch[1]})
    print "test accuracy %g"%sess.run(accuracy, feed_dict={
    x: mnist.test.images, y_: mnist.test.labels})

  sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})


saver = tf.train.Saver([W_1, b_1, W_2,b_2])
save_path = saver.save(sess, "./model.ckpt")
print "Model saved in file: ", save_path


#print "test accuracy %g"%sess.run(accuracy, feed_dict={
#    x: mnist.test.images, y_: mnist.test.labels})


##producing data to get the plots we like

#f = open('nnout.dat', 'w')
#output of neural net
#ii=0
#for i in range(Ntemp):
#  av=0.0
#  for j in range(samples_per_T):
#        batch=(mnist.test.images[ii,:].reshape((1,lx*lx)),mnist.test.labels[ii,:].reshape((1,numberlabels)))     
#        res=sess.run(y_conv,feed_dict={x: batch[0], y_: batch[1]})
#        av=av+res 
#        #print ii, re
         
#        ii=ii+1 
#  av=av/samples_per_T
#  f.write(str(i)+' '+str(av[0,0])+' '+str(av[0,1])+"\n")  
#  #print i,av   
#f.close()       

## accuracy vs temperature
#f = open('acc.dat', 'w')
#for ii in range(Ntemp):
#  batch=(mnist.test.images[ii*samples_per_T:ii*samples_per_T+samples_per_T,:].reshape(samples_per_T,lx*lx), mnist.test.labels[ii*samples_per_T:ii*samples_per_T+samples_per_T,:].reshape((samples_per_T,numberlabels)) )
#  train_accuracy = sess.run(accuracy,feed_dict={
#        x:batch[0], y_: batch[1]}) 
#  f.write(str(ii)+' '+str(train_accuracy)+"\n") #
  #  print ii, train_accuracy


#f.close()





