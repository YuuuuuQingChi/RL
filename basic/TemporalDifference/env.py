class CliffWalkingEnv:
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol
        self.nrow = nrow
        self.reset()

    def step(self, action):
        change = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        dx, dy = change[action]
        self.x = min(self.ncol - 1, max(0, self.x + dx))
        self.y = min(self.nrow - 1, max(0, self.y + dy))

        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False

        if self.y == self.nrow - 1 and self.x > 0:
            done = True
            if self.x != self.ncol - 1:
                reward = -100

        return next_state, reward, done

    def reset(self):
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x
