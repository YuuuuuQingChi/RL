import environment
import policy
import copy
import view_policy

env = environment.CliffWalkingEnv()
action_meaning = ['上', '下', '左', '右']

theta = 0.001
gamma = 0.9
agent = policy.PolicyIteration(env=env,theta = theta,gamma= gamma)
agent.policy_iteration()
view_policy.print_agent(agent, action_meaning, list(range(37, 47)), [47])
