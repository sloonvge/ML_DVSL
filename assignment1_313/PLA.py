import numpy as np

class PLA(object):
    
	def __init__(self):
		pass
		
	def _sign(self, x):#符号函数
		x[np.where(x >= 0)] = 1
		x[np.where(x < 0)] = -1
		return x
	
	def pred(self, X, y, W):#用于计算在指定的W下的正确率
		return (self._sign(X.dot(W)) == y).mean()
	
	def train(self, X, y, W, num_iters=2000):
		'''
		X: n * d维
		y: n * 1维
		W: d * 1维
		num_iters: 迭代次数
		'''
		self._W = W
		Ws = []#用来存储W的历史值
		#acc_best = 0 
		for t in range(num_iters+1):
			i = np.random.randint(0, len(X)-1)#随机选取一个数据
			if t % (num_iters / 10) == 0:
				Ws.append((t, self._W[:, 0].tolist()))
			if y[i] * self._sign(X[i:i+1,:].dot(self._W)) <= 0:#判断是否预测错误
				self._W = self._W + (y[i] * X[i]).reshape(-1, 1)#更新W值
		return self._W, Ws
    
	def test(self, X, y):#用于判断每一次W更新后的正确率
		return self.pred(X, y, self._W)
		

class Pocket(PLA):
    
	def train(self, X, y, W, num_iters=2000):
		'''
		X: n * d维
		y: n * 1维
		W: d * 1维
		num_iters: 迭代次数
		'''
		self._W = W
		Ws = []
		acc_best = 0
		W_best = None#记录在训练集上表现最好的W
		for t in range(num_iters+1):
			i = np.random.randint(0, len(X)-1)
			if y[i] * self._sign(X[i:i+1,:].dot(self._W)) <= 0:#判断是否预测错误，如果错误
				acc_temp = (((self.test(X, y))) * 100)#就先计算当前的正确率
				if acc_temp > acc_best:#如果当前W的表现优于最好的W
					acc_best = acc_temp#更新最好的准确率
					W_best = self._W#以及对应的W
					Ws.append((t, self._W[:, 0].tolist()))#
				else:#如果当前W的表现没有优于最好的W就更新W的值
					self._W = self._W + (y[i] * X[i]).reshape(-1, 1)
		#两个返回值：在测试集上表现最好的W和一个元素是W的历史最好值的列表
		return W_best, Ws
    

		
