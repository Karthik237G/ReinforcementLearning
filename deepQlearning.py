import os
import numpy as np
import tensorflow as tf
class deepQnetwork(object):
    def __init__(self,lr,n_actions,name,fcl_dims=256,input_dims=(210,160,4),chkpt_dir='tmp/dqn'):
        self.lr=le
        self.n_actions=n_actions
        self.name=name
        self.fcl_dims=fcl_dims
        slef.input_dims=input_dims
        self.sess=tf.Session()
        self.build_network()
        self.sess.run(tf.global_variables_initializer())
        self.saver=tf.train.Saver()
        self.checkpoint_file=os.path.join(chkpt_dir,'deepQnet.ckpt')
        self.params=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=self.name)
        
    def build_net(self):
        with tf.variable_scope(self.name):
            self.input=tf.placeholder(tf.float32,shape=[None,*self.input_dims],name='inputs')
            self.actions=tf.placeholder(tf.float32,shape=[None,self.n_actions],name='action_taken')
            self.q_target=tf.placeholder(tf.float32,shape=[None,self.input_dims])
            conv1=tf.layers.conv2d(inputs=self.input,filtres=32,kernal_size=(8,8),strides=4,name='conv1',kernal_initializer=tf.variance_scaling_initializer(scale=2))            
            conv1_activated=tf.nn.relu(conv1)
            conv2=tf.layers.conv2d(inputs=conv1_activated,filtres=64,kernal_size=(4,4),strides=2,name='conv2',kernal_initializer=tf.variance_scaling_initializer(scale=2))
            conv2_activated=tf.nn.relu(conv2)
            conv3=tf.layers.conv2d(inputs=conv2_activated,filtres=128,kernal_size=(3,3),strides=1,name='conv3',kernal_initializer=tf.variance_scaling_initializer(scale=2))
            conv3_activated=tf.nn.relu(conv3)
            flat=tf.layers.flatten(conv3_activated)
            dense1=tf.layer.dense(flat,units=self.fcl_dims,activation=tf.nn.relu,kernal_initializer=tf.variance_scaling_initializer(scale=2))
            self.Q_values=tf.layer.dense(flat,units=self.n_actions,kernal_initializer=tf.variance_scaling_initializer(scale=2))
            self.q=tf.reduce_sum(tf.multiply(self.Q_values,self.actions))
            self.loss=tf.reduce_mean(tf.square(self.q-self.q_target))
            self.train_op=tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def load_checkpoint(self):
        print('loadingcheckpoint')
        self.saver.restore(self.sess,self.checkpoint_file)

    def save_checkpoint(self):
        print('saving')
        self.saver.save(self.sess,self.checkpoint_file)
class agent(object):
    def __init__(self,alpha,gamma,mem_size,n_actions,epsilon,batch_size,replace_target=5000,input_dims=(210,160,4),q_next='tmo/q_next',q_eval='tmp/q_eval'):
        self.n_actions=n_actions
        self.action_space=[ i for i in range(self.n_actions)]
        self.gamma=gamma
        self.mem_size=mem_size
        self.epislon=epislon
        self.batch_size=batch_size
        self.mem_cntr=0
        self.replace_target=replace_target
        self.q_next=deepQnetwork(alpha,n_actions,input_dims=input_dims,name='q_next',chkpt_dir=q_next_dir)
        self.q_eval=deepQnetwork(alpha,n_actions,input_dims=input_dims,name='q_eval',chkpt_dir=q_eval_dir)
        self.state_memory=np.zeros((self.mem_size,*input_dims))
        self.new_state_memory=np.zeros((self.mem_size,*input_dims))
        self.action_memory=np.zeros((self.mem_size,self.n_actions))
        self.reward_memory=np.zeros(self.mem_size)
        self.terminal_memory=np.zeros(self.mem_size,dtype=np.int8)
    def store_transition(self,state,action,reward,state_,terminal):
        index=self.mem_cntr%self.mem_size
        self.state_memory[index]=state
        actions=np.zeros(self.n_actions)
        actions[action]=1.0
        self.action_memory[index]=actions
        self.action_memory[index]=reward
        self.new_state_memory[index]=state_
        self.terminal_memory[index]=terminal
        self.mem_cntr+=1

    def choose_action(self,state):
        rand=np.random.random()
        if rand<self.epsilon:
            action=np.random.choice(self.action_space)
        else:
            actions=self.q_eval.sess.run(self.q_eval.q_values,feed_dict={self.q_eval.input:state})
            action=np.argmax(actions)
        return action
    
    def lern(self):
        if self.mem_cntr%self.replace_target==:
            self.update_graph()
        max_mem=self.mem_cntr if self.mem_cntr<self.mem_size else self.mem_size
        
        
