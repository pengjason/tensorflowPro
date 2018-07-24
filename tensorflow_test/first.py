import tensorflow as tf


def get_tf_info():
    print(tf.__version__)
    print(tf.__path__)

def matmultest():

    v1 = tf.constant([[2,3]])
    v2 = tf.constant([[2],[3]])
    
    product = tf.matmul(v1,v2)
    print(product)
    
    sess = tf.Session()
    result = sess.run(product)
    print(result)
    sess.close()
    
    
def init_for_loop():
    num = tf.Variable(0,name="count")
    new_value = tf.add(num,10)
    op = tf.assign(num,new_value)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('init value:',sess.run(num))
        for i in range(5):
            sess.run(op)
            print(sess.run(num))
   
def feed_placeholder():
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    new_value = tf.multiply(input1, input2)
    with tf.Session( ) as sess:
        print('get feed placeholder')
        print(sess.run(new_value,feed_dict={input1:23.0,input2:11.0}))

# print(tf.__version__)
# print(tf.__path__)
   
init_for_loop()
feed_placeholder()
print('test tensorflow')