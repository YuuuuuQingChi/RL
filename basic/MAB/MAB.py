import numpy as np
import matplotlib.pyplot as plt




class BernulliBandit:  # 多臂老虎机生成器
    def __init__(self, K):
        self.probabilities = np.random.uniform(
            0, 1, size=K
        )  # 指定区间 [low, high) 内的随机浮点数（注意是左闭右开区间）

        self.best_idx = np.argmax(
            self.probabilities
        )  # argmax（argument of the maximum，最大值的索引）
        self.best_probability = self.probabilities[self.best_idx]

        self.K = K

    def lottery(self, k):  # 抽奖函数
        if k <= self.K:
            if np.random.uniform(0, 1) < self.probabilities[k]:
                return 1
            else:
                return 0


# test = BernulliBandit(19)
# example_once = test.lottery(10)
# print(example_once)
# print(test.best_idx,test.best_probability)


class Solver:
    """多臂老虎机算法基本框架"""

    def __init__(self, bandit: BernulliBandit):
        self.bandit = bandit  # 获得多臂老虎机生成器
        self.counts = np.zeros(self.bandit.K)
        self.regret = 0
        self.actions = []  # 存储每一次的action和regret
        self.regrets = []

    def update_regret(self, k):
        # 计算累积懊悔并保存,k为本次动作选择的拉杆的编号
        self.regret += self.bandit.best_probability - self.bandit.probabilities[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        raise NotImplementedError

    def run(self, num_steps):
        # 运行一定次数,num_steps为总运行次数
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)

def plot_results(solvers: Solver, solver_names):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel("Time steps")
    plt.ylabel("Cumulative regrets")
    plt.title("%d-armed bandit" % solvers[0].bandit.K)
    plt.legend()
    plt.show()

class EpsilonGreedy(Solver):
    def __init__(self, bandit, epsilon=0.5, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array(
            [init_prob] * self.bandit.K
        )  # 创建一个长度为 K 的 NumPy 数组\所有元素都初始化为 init_prob

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.lottery(k)
        self.counts[k] += 1
        self.estimates[k] += (r - self.estimates[k]) / self.counts[k]

        return k


class DecayingEpsilonGreedy(Solver):
    """epsilon值随时间衰减的epsilon-贪婪算法,继承Solver类"""

    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:  # epsilon值随时间衰减
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)

        r = self.bandit.lottery(k)
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (r - self.estimates[k])

        return k


K = 10
bandit_10_arm = BernulliBandit(K)
# np.random.seed(1)
# epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
# epsilon_greedy_solver.run(5000)
# print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)
# plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])


class UCB(Solver):
    def __init__(self, bandit: BernulliBandit, c, init_prob=1.0):
        super().__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.c = c

    def run_one_step(self):
        self.total_count += 1
        ucb = np.sqrt(np.log(self.total_count) / (2 * (self.counts + 1)))
        prb_ = self.estimates + self.c * ucb
        k = np.argmax(prb_)
        r = self.bandit.lottery(k)
        self.estimates[k] += (r - self.estimates[k]) / (self.counts[k] +1)
        return k


# decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
# decaying_epsilon_greedy_solver.run(5000)
# print("epsilon值衰减的贪婪算法的累积懊悔为：", decaying_epsilon_greedy_solver.regret)
# plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])



coef = 1.1  # 控制不确定性比重的系数
UCB_solver = UCB(bandit_10_arm, coef)
UCB_solver.run(5000)
print('上置信界算法的累积懊悔为：', UCB_solver.regret)
plot_results([UCB_solver], ["UCB"])