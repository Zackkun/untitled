import numpy as np
from matplotlib import colors
from sklearn import svm
from sklearn.svm import SVC
from sklearn import model_selection
import matplotlib.pyplot as plt
import matplotlib as mpl

#将数据里的类别字符串(bytes)转为整形，便于数据加载
def iris_type(s):
    irisname_change = {b'Iris-setosa':0,b'Iris-versicolor':1,b'Iris-virginica':2}
    return irisname_change[s]

#加载数据
data_path='E:/iris.data'
data = np.loadtxt(data_path,dtype=float,delimiter=',',converters={4:iris_type})
#print(data)
#print(data.shape)
#数据分割
x,y = np.split(data, (4,),axis=1)
x = x[:,0:4]
#print(x)
#样本集划分成训练和测试集
x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,random_state=1,test_size=0.3)

#svm分类器
def classifier():
    """
    C：惩罚项，float类型，可选参数，默认为1.0，C越大，即对分错样本的惩罚程度越大，因此在训练样本中准确率越高，但是泛化能力降低，也就是对测试数据的分类准确率降低。相反，减小C的话，容许训练样本中有一些误分类错误样本，泛化能力强。对于训练样本带有噪声的情况，一般采用后者，把训练样本集中错误分类的样本作为噪声。
    SVM中常用核函数一般是线性核函数和高斯核函数。以sklearn中的SVC，提供的’linear’和’rbf’做说明。面向[n,m]原始数据集，一般的选取准则：
    相对于n，m很大。比如m≥n, m=10000, n=10~1000,即(m/n)>10。考虑’linear’
    m很小，n一般大小。比如m=1-1000, n=10~10000,即(m/n)在[0.0001,100].考虑’rbf’
    m很小，n很大。比如n=1-1000，m=50000+，即(m/n)在[~,0.02].增加m的量，考虑’linear’
    """
    clf = svm.SVC(C=0.5,kernel='linear',decision_function_shape='ovr')
    return clf

#定义svm
clf = classifier()

#训练模型
def train(clf,x_train,y_train):
    clf.fit(x_train,y_train.ravel())

train(clf,x_train,y_train)

#测试模型
print(clf.score(x_train,y_train))
print(clf.score(x_test,y_test))
print(clf.predict(x_test))

def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    print('%s Accuracy:%.3f' %(tip, np.mean(acc)))

show_accuracy(clf.predict(x_test), y_test, 'testing data')

#数据可视化，4个变量两两比对
def draw(clf, x):
    iris_feature = 'sepal length', 'sepal width', 'petal lenght', 'petal width'
    # 开始画图
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围
    x3_min, x3_max = x[:, 2].min(), x[:, 2].max()  # 第2列的范围
    x4_min, x4_max = x[:, 3].min(), x[:, 3].max()  # 第3列的范围

    cm_dark = mpl.colors.ListedColormap(['g', 'b', 'r'])

    plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y), edgecolor='k', s=50, cmap=cm_dark)  # 样本点
    plt.scatter(x_test[:, 0], x_test[:, 1], s=120, facecolor='none', zorder=10)  # 测试点
    plt.xlabel(iris_feature[0], fontsize=20)
    plt.ylabel(iris_feature[1], fontsize=20)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title('svm in iris data classification', fontsize=30)
    plt.grid()
    plt.show()

    plt.scatter(x[:, 2], x[:, 3], c=np.squeeze(y), edgecolor='k', s=50, cmap=cm_dark)  # 样本点
    plt.scatter(x_test[:, 2], x_test[:, 3], s=120, facecolor='none', zorder=10)  # 测试点
    plt.xlabel(iris_feature[2], fontsize=20)
    plt.ylabel(iris_feature[3], fontsize=20)
    plt.xlim(x3_min, x3_max)
    plt.ylim(x4_min, x4_max)
    plt.title('svm in iris data classification', fontsize=30)
    plt.grid()
    plt.show()

    plt.scatter(x[:, 1], x[:, 2], c=np.squeeze(y), edgecolor='k', s=50, cmap=cm_dark)  # 样本点
    plt.scatter(x_test[:, 1], x_test[:, 2], s=120, facecolor='none', zorder=10)  # 测试点
    plt.xlabel(iris_feature[1], fontsize=20)
    plt.ylabel(iris_feature[2], fontsize=20)
    plt.xlim(x2_min, x2_max)
    plt.ylim(x3_min, x3_max)
    plt.title('svm in iris data classification', fontsize=30)
    plt.grid()
    plt.show()

    plt.scatter(x[:, 0], x[:, 2], c=np.squeeze(y), edgecolor='k', s=50, cmap=cm_dark)  # 样本点
    plt.scatter(x_test[:, 0], x_test[:, 2], s=120, facecolor='none', zorder=10)  # 测试点
    plt.xlabel(iris_feature[0], fontsize=20)
    plt.ylabel(iris_feature[2], fontsize=20)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x3_min, x3_max)
    plt.title('svm in iris data classification', fontsize=30)
    plt.grid()
    plt.show()

    plt.scatter(x[:, 0], x[:, 3], c=np.squeeze(y), edgecolor='k', s=50, cmap=cm_dark)  # 样本点
    plt.scatter(x_test[:, 0], x_test[:, 3], s=120, facecolor='none', zorder=10)  # 测试点
    plt.xlabel(iris_feature[0], fontsize=20)
    plt.ylabel(iris_feature[3], fontsize=20)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x4_min, x4_max)
    plt.title('svm in iris data classification', fontsize=30)
    plt.grid()
    plt.show()

    plt.scatter(x[:, 1], x[:, 3], c=np.squeeze(y), edgecolor='k', s=50, cmap=cm_dark)  # 样本点
    plt.scatter(x_test[:, 1], x_test[:, 3], s=120, facecolor='none', zorder=10)  # 测试点
    plt.xlabel(iris_feature[1], fontsize=20)
    plt.ylabel(iris_feature[3], fontsize=20)
    plt.xlim(x2_min, x2_max)
    plt.ylim(x4_min, x4_max)
    plt.title('svm in iris data classification', fontsize=30)
    plt.grid()
    plt.show()

draw(clf,x)