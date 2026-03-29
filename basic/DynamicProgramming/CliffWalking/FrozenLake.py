import gym
import policy
import view_policy
env = gym.make("FrozenLake-v1")  # 创建环境
env = env.unwrapped  # 解封装才能访问状态转移矩阵P
env.render()  # 环境渲染,通常是弹窗显示或打印出可视化的环境

print("P的大小: 状态数=%d, 每个状态的动作数=%d" % (len(env.P), len(env.P[0])))

holes = set()
ends = set()
for s in env.P:
    for a in env.P[s]:
        for s_ in env.P[s][a]:
            if s_[2] == 1.0:  # 获得奖励为1,代表是目标
                ends.add(s_[1])
            if s_[3] == True:
                holes.add(s_[1])
holes = holes - ends
print("冰洞的索引:", holes)
print("目标的索引:", ends)

action_meaning = ['左', '下', '右', '上']

theta = 0.001
gamma = 0.9
agent = policy.PolicyIteration(env=env,theta = theta,gamma= gamma)
agent.policy_iteration()
view_policy.print_agent(agent,action_meaning,holes,ends)
