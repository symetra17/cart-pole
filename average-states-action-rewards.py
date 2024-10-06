# state-action0-action1-action2-action3-reward model
# not just old school state-action-reward model

import os
#os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["KERAS_BACKEND"] = "jax"
#os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import keras
from keras import ops
from keras import layers
import tensorflow as tf
import gymnasium as gym
from icecream import ic
import time
import random
import matplotlib.pyplot as plt
import glob
import uuid
import cv2 as cv

MAX_REW = 150
ACT_LEN = 7


np_seq_set_7 = np.array([
        [0,0,0,1,0,1,1],        [0,0,0,1,1,0,0], 
        [0,0,0,1,1,0,1],        [1,0,1,1,0,0,1], 
        [1,1,1,0,1,0,1],        [1,1,1,0,1,1,0], 
        [1,1,1,0,1,1,1],        [1,1,1,1,0,0,0], 
        [1,1,1,1,0,0,1],        [1,0,1,1,0,1,0],
        [1,0,1,1,0,1,1],        [1,0,1,1,1,0,0], 
        [1,0,1,1,1,0,1],        [1,0,1,1,1,1,0], 
        [1,0,1,1,1,1,1],        [1,1,0,0,0,0,0], 
        [0,0,0,1,0,0,0],        [0,0,0,1,0,0,1], 
        [0,0,0,1,0,1,0],        [1,1,0,0,0,0,1], 
        [1,1,0,0,0,1,0],        [1,1,0,0,0,1,1], 
        [1,1,0,0,1,0,0],        [1,1,0,0,1,0,1], 
        [1,1,0,0,1,1,0],        [1,1,0,0,1,1,1],
        [1,1,0,1,0,0,0],        [1,1,0,1,0,0,1], 
        [1,1,0,1,0,1,0],        [1,1,0,1,0,1,1],
        [1,1,0,1,1,0,0],        [1,1,0,1,1,0,1], 
        [1,1,0,1,1,1,0],        [1,1,0,1,1,1,1],
        [1,1,1,0,0,0,0],        [1,1,1,0,0,0,1], 
        [0,1,0,0,0,0,0],        [0,1,0,0,0,0,1], 
        [0,1,0,0,0,1,0],        [1,1,1,0,0,1,0], 
        [1,1,1,0,0,1,1],        [1,1,1,0,1,0,0], 
        [0,0,0,0,0,0,0],        [0,0,0,0,0,0,1], 
        [0,0,0,0,0,1,0],        [0,0,0,0,0,1,1], 
        [0,0,1,1,0,1,0],        [0,0,1,1,0,1,1], 
        [0,0,1,1,1,0,0],        [0,0,1,1,1,0,1], 
        [0,0,1,1,1,1,0],        [0,0,1,1,1,1,1],
        [0,1,0,0,0,1,1],        [0,1,0,0,1,0,0], 
        [0,1,0,0,1,0,1],        [0,1,0,0,1,1,0], 
        [0,1,0,0,1,1,1],        [0,1,0,1,0,0,0], 
        [0,1,0,1,0,0,1],        [0,1,0,1,0,1,0], 
        [0,1,0,1,0,1,1],        [0,1,0,1,1,0,0], 
        [0,1,0,1,1,0,1],        [0,1,0,1,1,1,0], 
        [0,1,0,1,1,1,1],        [0,1,1,0,0,0,0], 
        [0,1,1,0,0,0,1],        [0,1,1,0,0,1,0], 
        [0,1,1,0,0,1,1],        [0,1,1,0,1,0,0], 
        [0,1,1,0,1,0,1],        [0,1,1,0,1,1,0], 
        [0,1,1,0,1,1,1],        [0,1,1,1,0,0,0], 
        [0,1,1,1,0,0,1],        [0,1,1,1,0,1,0], 
        [0,1,1,1,0,1,1],        [0,1,1,1,1,0,0], 
        [0,1,1,1,1,0,1],        [0,1,1,1,1,1,0], 
        [0,1,1,1,1,1,1],        [1,0,0,0,0,0,0], 
        [1,0,0,0,0,0,1],        [1,1,1,1,0,1,0], 
        [1,1,1,1,0,1,1],        [1,1,1,1,1,0,0], 
        [1,1,1,1,1,0,1],        [1,1,1,1,1,1,0], 
        [1,1,1,1,1,1,1],        [1,0,0,0,0,1,0], 
        [1,0,0,0,0,1,1],        [1,0,0,0,1,0,0], 
        [1,0,0,0,1,0,1],        [1,0,0,0,1,1,0], 
        [1,0,0,0,1,1,1],        [1,0,0,1,0,0,0], 
        [1,0,0,1,0,0,1],        [1,0,0,1,0,1,0], 
        [1,0,0,1,0,1,1],        [1,0,0,1,1,0,0], 
        [1,0,0,1,1,0,1],        [1,0,0,1,1,1,0], 
        [1,0,0,1,1,1,1],        [1,0,1,0,0,0,0], 
        [1,0,1,0,0,0,1],        [1,0,1,0,0,1,0], 
        [0,0,0,0,1,0,0],        [0,0,0,0,1,0,1], 
        [0,0,0,0,1,1,0],        [0,0,0,0,1,1,1],
        [0,0,0,1,1,1,0],        [0,0,0,1,1,1,1],
        [0,0,1,0,0,0,0],        [0,0,1,0,0,0,1], 
        [0,0,1,0,0,1,0],        [0,0,1,0,0,1,1], 
        [0,0,1,0,1,0,0],        [0,0,1,0,1,0,1], 
        [0,0,1,0,1,1,0],        [0,0,1,0,1,1,1],
        [0,0,1,1,0,0,0],        [0,0,1,1,0,0,1], 
        [1,0,1,0,0,1,1],        [1,0,1,0,1,0,0], 
        [1,0,1,0,1,0,1],        [1,0,1,0,1,1,0], 
        [1,0,1,0,1,1,1],        [1,0,1,1,0,0,0], 
           ])     # 128x7


def get_s4ar_model():
    # average state-action-reward model
    # state-action0-action1-action2-action3-reward model
    i = keras.layers.Input(shape=(4,))
    i2 = keras.layers.Input(shape=(ACT_LEN,))
    i2 -= 0.5
    o = keras.layers.Concatenate()([i,i2])
    o = keras.layers.Dense(128, activation='gelu')(o)
    o = keras.layers.Dense(64, activation='gelu')(o)
    o = keras.layers.Dense(1, activation=None)(o)
    model = keras.Model(inputs=[i,i2], outputs=o)
    model.compile(loss=keras.losses.Huber(), 
              optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              metrics=['mae'])
    return model


def policy(init_state, model):
    # search demo
    ncombi = np.power(2,ACT_LEN)
    np_state = np.zeros(shape=(ncombi,4), dtype=np.float32)
    env_s = np.array(init_state)
    np_state[:,:] = env_s
    predi = model([np_state,np_seq_set_7])
    idx = np.argmax(predi)
    return int(np_seq_set_7[idx,0]), np.max(predi)


#def train_s4ar():         # actor model
#    files = glob.glob('s4ar-data-*.npy')
#    random.shuffle(files)
#    np_states = None
#    for fn in files:
#        combo = np.load(fn)
#        if np_states is None:
#            np_states  = combo[:,0:4].copy()
#            np_actions = combo[:,4:4+ACT_LEN].copy()
#            np_rewards = combo[:,4+ACT_LEN:4+ACT_LEN+1].copy()
#        else:
#            np_states  = np.concatenate([np_states,  combo[:,0:4]], axis=0)
#            np_actions = np.concatenate([np_actions, combo[:,4:4+ACT_LEN]], axis=0)
#            np_rewards = np.concatenate([np_rewards, combo[:,4+ACT_LEN:4+ACT_LEN+1]], axis=0)
#
#    model = get_s4ar_model()
#    try:
#        model.load_weights('s4ar.weights.h5')
#    except:
#        pass
#    
#    model.fit([np_states, np_actions], np_rewards,
#                  batch_size=1000, 
#                  epochs=100)
#    model.save_weights('s4ar.weights.h5')



def train(model):
    files = glob.glob('s4ar-data-*.npy')
    random.shuffle(files)
    np_states = None
    for fn in files:
        combo = np.load(fn)
        if np_states is None:
            np_states  = combo[:,0:4].copy()
            np_actions = combo[:,4:4+ACT_LEN].copy()
            np_rewards = combo[:,4+ACT_LEN:4+ACT_LEN+1].copy()
        else:
            np_states  = np.concatenate([np_states,  combo[:,0:4]], axis=0)
            np_actions = np.concatenate([np_actions, combo[:,4:4+ACT_LEN]], axis=0)
            np_rewards = np.concatenate([np_rewards, combo[:,4+ACT_LEN:4+ACT_LEN+1]], axis=0)

    np_rewards -= np_rewards.mean()
    np_rewards /= np_rewards.std()
    model.fit([np_states, np_actions], np_rewards, batch_size=1000, 
              epochs=10)
    model.save_weights('s7ar.weights.h5')




if __name__=='__main__':
    display     = 0
    use_policy  = 1
    exploration = 1
    step_cnt_slow = 8.0

    if display:
        env = gym.make("CartPole-v1", render_mode='human',)
    else:
        env = gym.make("CartPole-v1")

    model = get_s4ar_model()
    model.load_weights('s7ar.weights.h5')
    
    img = np.zeros(shape=(300,1000), dtype=np.uint8)

    state_list = []
    action_list = []
    reward_list = []

    for g in range(2000):
        env.reset()
        env.state[0] = np.random.uniform(-1.9, 1.9)
        env.state[1] = np.random.uniform(-1.0, 1.0)
        env.state[2] = np.random.uniform(-0.13, 0.13)
        env.state[3] = np.random.uniform(-0.1, 0.1)
        episo_states = []
        episo_actions = []
        img[:] = 0
        s = np.array(env.state)
        for t in range(301):
            episo_states.append(s.copy())
            a, init_pred = policy(s, model)
            if exploration:
                if random.choice([1,0,0,0,0,0,0,0]):
                    a = int(np.random.randint(0,2))
            episo_actions.append(a)
            s_next,_,fell,_,_ = env.step(a)
            s[:] = s_next[:]
            if fell:
                break

            if display:       # state space display
                x = s_next[0]
                x *= (img.shape[1]/2)/2.4
                theta = s_next[2]
                theta *= 150.0/0.21
                cv.circle(img, (round((img.shape[1]/2)+x), 150+round(theta)), 2, 
                          (198), -1, cv.LINE_AA)
                cv.imshow('', img)
                cv.waitKey(10)
                
        if display:
            ic(t)

        step_cnt_slow = 0.95*step_cnt_slow+0.05*t
        print('Episode  ', g, '  buf len ', len(state_list),  '  Average ',int(step_cnt_slow))
        
        if fell:
            for n in range(t+1):
                state_list.append(episo_states[n])
                eacts = episo_actions[n:n+ACT_LEN]
                if len(eacts) < ACT_LEN:
                    rand_shit = np.random.randint(0,2,size=(ACT_LEN-len(eacts)))
                    rand_shit = list(rand_shit)
                    eacts.extend(rand_shit)
                action_list.append(eacts)
                epr = min(MAX_REW, t-n)
                discnt_epr = 0
                for h in range(epr):
                    discnt_epr += np.power(0.99,h)
                reward_list.append(discnt_epr)
    
        
        if len(state_list) >= 1000:
            np_states = np.vstack(state_list)
            np_actions = np.vstack(action_list)
            np_reward = np.vstack(reward_list)
            state_list.clear()
            action_list.clear()
            reward_list.clear()
            np.save('s4ar-data-%s.npy'%str(uuid.uuid4()), 
                    np.concatenate([np_states, np_actions, np_reward], axis=1))
            print('s4ar data saved')            
            train(model)
            