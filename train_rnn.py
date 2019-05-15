# -*- coding:utf-8 -*-
import logging
import time
import tensorflow as tf
from build_rnn import AFD_RNN
from utils import parser_cfg_file
from data_load import DataLoad

class AFD_RNN_Train(object):

    def __init__(self, train_config):

        self.learing_rate = float(train_config['learning_rate'])
        self.train_iterior = int(train_config['train_iteration'])
        self.cover_train = train_config['cover_train'] == 'True'
        self._train_logger_init()#为训练类初始化logger
        net_config = parser_cfg_file('./config/rnn_net.cfg')# 设置了两个config，
                                                            # train.cfg负责对训练的时间等等粗粒度的控制,
                                                            # net.cfg负责对RNN的超参进行设置
        self.rnn_net = AFD_RNN(net_config)
        self.predict = self.rnn_net.build_net_graph() #对网络进行初始化，定义网络输出为变量self.predict

      #  self.predict =  self.predict (name = 'predict')

        self.label = tf.placeholder(tf.float32, [None, self.rnn_net.time_step, self.rnn_net.class_num])#设置变量label

    def _compute_loss(self):
        with tf.name_scope('loss'):#真的就是命名
            # [batchszie, time_step, class_num] ==> [time_step][batchsize, class_num]
            predict = tf.unstack(self.predict, axis=0) # [64,11]
            label = tf.unstack(self.label, axis=1) #[,11]

            loss = [tf.nn.softmax_cross_entropy_with_logits(labels=label[i], logits=predict[i]) for i in range(self.rnn_net.time_step) ]
            loss = tf.reduce_mean(loss)
            train_op = tf.train.AdamOptimizer(self.learing_rate).minimize(loss)
        return loss, train_op

    def train_rnn(self):

        loss, train_op = self._compute_loss() #认为train_op是用以sess.run的时候调用的

        with tf.name_scope('accuracy'):
            predict = tf.transpose(self.predict, [1,0,2]) #对预测的x y进行转置
#            print(predict.shape())
            correct_pred = tf.equal(tf.argmax(self.label, axis=2), tf.argmax(predict, axis=2))#返回True 和 False 组成的张量
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))# 这里tf.cast把True，False转换成1.0，0.0的float32形式

        dataset = DataLoad('./dataset/train/', time_step=self.rnn_net.time_step, class_num= self.rnn_net.class_num)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            if not self.cover_train:
                ckpt = tf.train.get_checkpoint_state('./model/')
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Module restore...')

            for step in range(1, self.train_iterior+1):
                x, y = dataset.get_batch(self.rnn_net.batch_size)
                if step == 1:
                    feed_dict = {self.rnn_net.input_tensor: x, self.label: y}
                else:
                    # 只要不是首次训练，state会不断传回到cell_state上
                    feed_dict = {self.rnn_net.input_tensor: x, self.label: y, self.rnn_net.cell_state:state}
                
                # state 会不断在每次训练过程中传出来再传进去
                _, compute_loss, state = sess.run([train_op, loss, self.rnn_net.cell_state], feed_dict=feed_dict)

                if step%10 == 0:
                    compute_accuracy = sess.run(accuracy, feed_dict=feed_dict)
                    self.train_logger.info('train step = %d,loss = %f,accuracy = %f'%(step, compute_loss, compute_accuracy))
                if step%100 == 0:
                    save_path = saver.save(sess, './model/model.ckpt')
                    self.train_logger.info("train step = %d ,model save to =%s" % (step, save_path))

    def _train_logger_init(self):#这个函数初始化的时候调用，所以以_开头
        """
        初始化log日志
        :return:
        """
        self.train_logger = logging.getLogger('train') #创建一个以“train”命名的logger
        self.train_logger.setLevel(logging.DEBUG)#设置这个logger的级别以确定什么优先级的信息需要被log下来

        # 添加文件输出
        log_file = './train_logs/' + time.strftime('%Y%m%d%H%M', time.localtime(time.time())) + '.logs'
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
        file_handler.setFormatter(file_formatter)
        self.train_logger.addHandler(file_handler)

        # 添加控制台输出
        consol_handler = logging.StreamHandler()
        consol_handler.setLevel(logging.DEBUG)
        consol_formatter = logging.Formatter('%(message)s')
        consol_handler.setFormatter(consol_formatter)
        self.train_logger.addHandler(consol_handler)

if __name__ == '__main__':
    train_config = parser_cfg_file('./config/train.cfg')
    train = AFD_RNN_Train(train_config)

    train.train_rnn()

    # a = tf.zeros([1,2,3])

    # b = tf.unstack(a, axis=1)
    # c = tf.zeros([2,1,3])
    # sess = tf.Session()
    # d = b[0]
    # print(sess.run(b[0]))
    #
    # pass