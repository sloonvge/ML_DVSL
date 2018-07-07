import numpy as np

class Features(object):#用于特征提取
    
	def __init__(self, X):
		self.X = X
	#提取平均亮度    
	def mean(self):
		X_mb = np.sum(self.X, axis=1)
		num = np.sum(self.X != 0, axis=1)
		return X_mb / num
    #提取上下对称性
	def upDown(self):
		X_ud = np.abs(self.X - self.X[:, ::-1])
		X_ud = np.sum(X_ud, axis=1)
		return X_ud
	
def featurenormal(X):#特征归一化
    std = np.std(X, axis=0)
    mean = np.mean(X, axis=0)
    X_change = (X - mean) / std
    return X_change, mean, std
	
def creatData(X, y, p=1, n=5):#X：数据 y：数据的label p:变换后数据对应label为1 n: 对应label为-1 
    Xp = X[np.where(y==p)[0]]
    Xn = X[np.where(y==n)[0]]
    Xpf = Features(Xp)
    Xnf = Features(Xn)
    matp = np.concatenate((Xpf.mean().reshape(-1, 1), Xpf.upDown().reshape(-1, 1)), axis=1)#Np * 2维
    matn = np.concatenate((Xnf.mean().reshape(-1, 1), Xnf.upDown().reshape(-1, 1)), axis=1)#Nn * 2维
    X_tr = np.concatenate((matp, matn), axis=0)#(Np + Nn) * 2维
    X_tr = featurenormal(X_tr)[0]#归一化
    X_train = np.c_[np.ones((X_tr.shape[0], 1)), X_tr]#(Np + Nn) * 3维
    y_train = np.r_[y[y==p], y[y==n]].reshape(-1, 1)
    y_train = y_train.astype(np.int8)#有符号整型
    y_train[np.where(y_train==p)] = 1
    y_train[np.where(y_train==n)] = -1
	
    return X_train, y_train
	