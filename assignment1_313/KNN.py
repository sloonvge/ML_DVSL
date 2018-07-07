import numpy as np
import matplotlib.pyplot as plt


class KNN(object):
    
    def __init__(self):
        pass
    
    def train(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def test(self, X, K=5):
        self.X = X#记录原始输入图片
        self.K_value = K#记录K值,用于画图
        if K < 8:#如果K值小于8，赋值为8，用于显示8个最近的曼哈顿距离对应的图片
            self.K = 8
        else:
            self.K = K
        self.y_pred = np.zeros(X.shape[0])#用于储存预测的标签
        self.y_pred_images = np.zeros((X.shape[0], self.K * X.shape[1]))#用于储存8个最近的曼哈顿距离对应的图
        self.y_pred_dists = np.zeros((X.shape[0], self.K))#用于储存8个最近的曼哈顿距离
        Md_dists = np.zeros((X.shape[0], self.X_train.shape[0]))#计算曼哈顿距离
        '''
        test_sum = np.sum(X**2, axis=1).reshape(-1, 1)
        train_sum = np.sum(self.X_train**2, axis=1).reshape(1, -1)
        L2_dists = test_sum - 2*X.dot(self.X_train.T) + train_sum
        L2_dists = np.sqrt(L2_dists)
        index = np.argsort(L2_dists, axis=1)
        select = index[:, :self.K]
        '''
        for i in range(X.shape[0]):#对输入的每一张图片操作
            Md_dists[i] = np.sum(np.abs(X[i] - self.X_train), axis=1)
            index = np.argsort(Md_dists[i])#对距离从小到大排序
            select = index[:self.K]#取出最近的self.K个距离（K < 8时，取出8个）
            nearest = []#用于存储与该图self.K张最近距离图片的标签（K < 8时，取8）
            nearest_images = np.zeros(self.K * X.shape[1])#用于存储与该图最近的K张图（K < 8时，取8）
            nearest_dists = np.zeros(self.K)#用于存储与该图的self.K个最近距离（K < 8时，取8）
            nearest = list(self.y_train[select[:K]])#从self.K个最近距离取出K个最近距离
            nearest_images = list(self.X_train[select, :].ravel())#
            nearest_dists = Md_dists[i, select]#
            most_nearest = max(set(nearest), key=nearest.count)#找出K张最近距离图片标签里出现次数最多的那个标签
            #np.argmax(np.bincount(nearest))
            self.y_pred[i] = most_nearest 
            self.y_pred_images[i] = nearest_images
            self.y_pred_dists[i] = nearest_dists
        '''
        在函数的返回值里取出K列，无论 self.K 的值是多少，最后都会返回K个最近的曼哈顿距离以及对应的图像
        '''
        return self.y_pred.astype(np.uint8), self.y_pred_images[:, :K * X.shape[1]], self.y_pred_dists[:, :K]
    
    def show(self, N=8):#默认显示8张距离最近的图片
        '''
        显示函数基于test的返回值作图，如果K < 8，不能选出最近的8张图（最多K张），所以，用self.x, 
        self.K_value, self.y_pred, self.y_pred_images, self.y_pred_dists储存有关数据。
        '''
        if self.X.shape[0] > 10 :#如果测试图片的数量大于10张，随机选出10张图片显示
            print('   Too many pictures to show, so we select 10 random pictures to show')
            index = np.random.choice(range(self.X.shape[0]), size=10)#随机确定10张图片的索引
            y_pred_images = self.y_pred_images[index]#根据上面的索引确定与预测图片最近的图片
            X = self.X[index]#根据索引确定随机选取的10张原始图片
            y_pred = self.y_pred.astype(np.uint8)[index]#根据索引确定预测的标签
        else:
            y_pred_images = self.y_pred_images
            X = self.X
            y_pred = self.y_pred.astype(np.uint8)
        #把图片放置在一个大矩阵里显示
        H, W = y_pred_images.shape
        cut = int(W / self.K * N)
        image8 = y_pred_images[:, :cut]
        W = int((image8.shape[1] / N))
        P = int(np.sqrt(W))
        imgmat = np.zeros((H * P, (N + 1) * P))#矩阵大小:(H * p , M * p) H:测试图片数量 p:图像大小 28
        for i in range(H):#												 M: 9 (1张原图+8张距离最近的图)
            a = image8[i].reshape( N, int(image8.shape[1] / N))
            imgmat[i * P:(i+1) * P, :P] = X[i].reshape(P, P)#矩阵首列放置原始图片
            for j in range(1, N + 1):
                imgmat[i * P:(i+1) * P, j * P:(j + 1) * P] = a[j-1].reshape(P, P)#矩阵其他列放置每一张图对应的最近图片
        plt.gray()
        plt.figure(figsize=(5, 5))
        plt.imshow(imgmat)
        plt.xlabel(str(self.K_value) + ' Nearst Neighbors')
        plt.ylabel(' '.join(str(i)+'   ' for i in y_pred[::-1]), size=15)#放置对应的预测标签
        '''
        The first column are the test images, and the numbers 
        in the left of the picture are the predict labels of the test images,
        the second to the last columns are the nearest to the nearer neighbors.
        '''
        plt.show()
        
    def showRGB(self, N=8):#显示RGB图像
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        if self.X.shape[0] > 10:
            print('   Too many pictures to show, so we select 10 random pictures to show')
            index = np.random.choice(range(self.X.shape[0]), size=10, )
            y_pred_images = self.y_pred_images[index]
            X = self.X[index]
            y_pred = self.y_pred.astype(np.uint8)[index]
        else:
            y_pred_images = self.y_pred_images
            X = self.X
            y_pred = self.y_pred.astype(np.uint8)
        H, W = y_pred_images.shape
        cut = int(W / self.K * N)
        image8 = y_pred_images[:, :cut]
        W = int((image8.shape[1] / (3 * N)))
        P = int(np.sqrt(W))
        imgmat = np.zeros((H * P, (N + 1) * P, 3)) 
        for d in range(3):
            for i in range(H):
                a = image8[i].reshape( N, int(image8.shape[1] / N))
                imgmat[i * P:(i+1) * P, :P, d] = X[i].reshape(P, P, 3)[:, :, d]
                for j in range(1, N + 1):
                    imgmat[i * P:(i+1) * P, j * P:(j + 1) * P, d] = a[j-1].reshape(P, P, 3)[:, :, d]
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(imgmat.astype(np.uint8))
        plt.xlabel(str(self.K_value) + ' Nearst Neighbors')
        plt.ylabel(' '.join(classes[i]+'    ' for i in y_pred[::-1]), size=15)
        '''
        The first column are the test images, and the words 
        in the left of the picture are the predict labels of the test images,
        the second to the last columns are the nearest to the nearer neighbors.
        '''
        plt.show()