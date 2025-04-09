import gym 

# 使用智能体控制小车移动
class SimpleAgent:
    def __init__(self,env):
        pass
    
    def decide(self,observation):
        '''决策'''
        position,velocity = observation
        lb = min(-0.09*(position+0.25)**2+0.03,
                 0.3*(position+0.9)**4-0.008)
        ub = -0.07*(position+0.38)**2+0.07
        
        if lb < velocity <ub:
            action = 2
        else:
            action = 0
        
        return action

    def learn(self,*args):
        '''学习'''
        pass

    '''现在还不是智能体，因为learn还没实现，需要在于环境的交互中学习'''
    
def play(env,agent,render=False,train=False):
    '''交互一回合'''
    '''params:
    env: 环境类
    agent: 智能体类
    render: 是否启用图形化界面
    train: 是否是训练模式
    '''
    episode_reward = 0 # 记录回合总奖励
    observation = env.reset()
    while True:
        if render:
            env.render()
        action = agent.decide(observation)
        next_observation,reward,done,indo = env.step(action)
        episode_reward += reward
        if train:
            agent.learn(observation,action,reward,done)
        if done:
            break
        observation = next_observation

    return episode_reward



if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    print(f"观测空间 = {env.observation_space}")
    print(f"动作空间 = {env.action_space}")
    print(f"观测范围 = {env.observation_space.low}~{env.observation_space.high}")
    print(f"动作数 = {env.action_space.n}")

    '''
    连续空间表示: gym.spaces.Discrete
    离散空间表示: gym.spaces.Box
    '''

    env.seed(3) # 设置随机种子，让结果可复现
    
    agent = SimpleAgent(env)
    
    # 对于小车上山任务，只要连续100个回合的平均奖励>-110，就认为该任务被解决了
    episode_reward = [play(env=env,agent=agent,render=True) for _ in range(100)]
    print(f"回合奖励={episode_reward}")
    env.close()