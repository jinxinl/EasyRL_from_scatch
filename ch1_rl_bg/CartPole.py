import pygame
import gym

if __name__=="__main__":
    env = gym.make("CartPole-v0") # 构建实验环境
    env.reset() # 重置一个回合
    for _ in range(1000):
        env.render() # 显式图形界面

        '''env.action_space.sample()会在所有动作空间中选一个输出'''
        action = env.action_space.sample() # 从动作空间中随机选取一个动作
        
        '''
        env.step()共四个返回值: 
            observation: 状态信息，返回结果会随游戏的不同而不同
            reward: 奖励值
            done: 游戏是否已完成，若完成需要重置游戏并开始新的回合
            info: 原始的用于诊断和调试的信息
        实现了 S -> A -> R -> S'
        '''
        observation,reward,done,info = env.step(action) # 提交动作
        print(observation) # 在本例中是四维观测 
    
    '''
    如果绘制了实验图形界面窗口，如果直接关闭窗口，可能会导致内存不能释放，甚至死机，
    关闭的最佳方式如下
    '''
    env.close() # 关闭环境



