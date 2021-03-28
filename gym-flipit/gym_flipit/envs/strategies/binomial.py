import random
from scipy.stats import norm
import math

class Binomial():
    def config(self,configs):
        self.beta = configs['beta']
        self.strategy = 'binomial'

    def first_move(self):
        return self.move(0)

    def move(self, LM):
        n = 1
        while random.randint(0,100)/100.0 > self.beta:
            n+=1
        return LM+n

    def pdf(self, x):
        return norm(self.delta+1,.1).pdf(x)

    def cdf(self, tau):
        return norm(self.delta,.1).cdf(tau)
