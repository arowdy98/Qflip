import random

class Custom():
    def config(self,configs):
        self.delta = configs['delta']
        self.dist = configs['dist']
        self.strategy = "custom"

    def first_move(self):
        return self.move(0)

    def move(self, LM):
        temp = random.random()
        cumulative = 0
        for i in range(self.delta):
            cumulative += self.dist[i]
            if cumulative > temp:
                mv = LM + i + 1
                break
        # print("Attacker:",mv)
        return mv

    def pdf(self,x):
        if x < self.delta:
            return self.dist[x]
        else:
            return 0

    def cdf(self,tau):
        cumulative = 0
        for i in range(self.delta):
            cumulative += self.dist[i]
            if cumulative > temp:
                return cumulative