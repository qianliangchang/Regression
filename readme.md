# <center>线性归回和Logistc回归原理分析及代码实现</center>
## 一、线性回归
### 1.原理介绍
`定义：`线性回归在假设特证满足线性关系，根据给定的训练数据训练一个模型，并用此模型进行预测。
为了了解这个定义，我们先举个简单的例子；我们假设一个线性方程 Y=2x+1, x变量为商品的大小，y代表为销售量；当月份x =5时，我们就能根据线性模型预测出 y =11销量；对于上面的简单的例子来说，我们可以粗略把 y =2x+1看到回归的模型；对于给予的每个商品大小都能预测出销量；当然这个模型怎么获取到就是我们下面要考虑的线性回归内容；并且在现实中影响销量（y）的因素好有很多，我们就拿商品大小（x₁)，商品价格为例 (x₂)为例:
在机器学习之前，获取数据是第一步（无米难巧妇之炊），假定我们的样本如下：其中x1 为商品的大小，x2 为商品的价格，y 为商品的销量:
| X1  | X2  |  Y  |
| --- | --- | --- |
|  3  |  5  | 10  |
|  2  |  6  | 14  |
|  6  | 10  | 28  |

### 2.模型推导
为了推导模型，在假设数据满足线性模型条件下，可以设定线性模型为;x1特征为商品的大小，X2特征为商品的价格
![20170331162541769](https://i.loli.net/2018/07/07/5b4067d43d58a.jpg)
#### 2.1 建模
``第一步：``
模型假定好后，我们把训练数据代入上面的设定模型中，可以通过模型预测一个样本最终值.
![20170331163451790](https://i.loli.net/2018/07/07/5b4068271c8d7.jpg)
然后样本真实值 y 和模型训练预测的值之间是有误差 $ε$ ,再假设训练样本的数据量很大的时候,根据中心极限定律可以得到 $∑ε$ 满足$(\mu ,δ^2)$高斯分布的；由于方程有截距项 ，故使用可以 $u =0$; 故满足$(0,δ^2)$的高斯分布；
![20170423220321946](https://i.loli.net/2018/07/07/5b40694cb859f.jpg)
如上面可知，对于每一个样本 x ,代入到 $p (y |x;	\theta)$ 都会得到一个y 的概率；又因为设定样本是独立同分布的；对其求最大似然函数：
![20170423222208668](https://i.loli.net/2018/07/07/5b4069f851e3d.jpg)
对其化简如下：
![20170423222746381](https://i.loli.net/2018/07/07/5b406a1e5ac4b.jpg)
以上就得到了回归的损失函数最小二乘法的公式，对于好多介绍一般对线性回归的线性损失函数就直接给出了上面的公式二乘法。
下面我们就对上面做了阶段性的总结：
线性回归，根据大数定律和中心极限定律假定样本无穷大的时候，其真实值和预测值的误差$\epsilon$的加和服从$(\mu,\delta)$的高斯分布且独立同分布，然后把$ε =y-\phi x$ 代入公式，就可以化简得到线性回归的损失函数；
#### 2.1 计算
``第二步：对损失函数进行优化也就是求出w,b，使的损失函数最小化；``
在这儿提供两种方法
方法一：使用矩阵（需要满足可逆条件）
![20170424105251818](https://i.loli.net/2018/07/07/5b406ed263cf9.jpg)
 以上就是按矩阵方法优化损失函数，但上面方法有一定的局限性，就是要可逆
方法二：梯度下降法
于梯度下降法的说明和讲解资料很多，深入的讲解这里不进行，可以参考(http://www.cnblogs.com/ooon/p/4947688.html)这篇博客，博主对梯度下降方法进行了讲解，我们这里就简单的最了流程解说;
![20170428135558922](https://i.loli.net/2018/07/07/5b40706279e7a.jpg)
总体流程就如上所示，就是求出每个变量的梯度；然后顺着梯度方向按一定的步长a,进行变量更新；下面我们就要求出每个变量的梯度，下面对每个θ进行梯度求解公式如下:
![20170428140502215](https://i.loli.net/2018/07/07/5b4070aee5d7f.jpg)
如上我们求出变量的梯度；然后迭代代入下面公式迭代计算就可以了：
![20170428141617416](https://i.loli.net/2018/07/07/5b4070d887266.jpg)
### 3.代码详情
按上面优化步骤就可以求出w,b,就可以获得优化的特征方程：说这么多先上个代码：
```python

#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import warnings
from sklearn.exceptions import  ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression,RidgeCV,LassoCV,ElasticNetCV
import matplotlib as mpl
import matplotlib.pyplot as plt

if __name__ == "__main__":

    warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
    np.random.seed(0)
    np.set_printoptions(linewidth=1000)
    N = 9
    x = np.linspace(0, 6, N) + np.random.randn(N)
    x = np.sort(x)
    y = x**2 - 4*x - 3 + np.random.randn(N)
    x.shape = -1, 1
    y.shape = -1, 1
    p =Pipeline([
        ('poly', PolynomialFeatures()),
        ('linear', LinearRegression(fit_intercept=False))])
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    np.set_printoptions(suppress=True)
    plt.figure(figsize=(8, 6), facecolor='w')
    d_pool = np.arange(1, N, 1)  # 阶
    m = d_pool.size
    clrs = []  # 颜色
    for c in np.linspace(16711680, 255, m):
        clrs.append('#%06x' % c)
    line_width = np.linspace(5, 2, m)
    plt.plot(x, y, 'ro', ms=10, zorder=N)
    for i, d in enumerate(d_pool):
        p.set_params(poly__degree=d)
        p.fit(x, y.ravel())
        lin = p.get_params('linear')['linear']
        output = u'%s：%d阶，系数为：' % (u'线性回归', d)
        print output, lin.coef_.ravel()
        x_hat = np.linspace(x.min(), x.max(), num=100)
        x_hat.shape = -1, 1
        y_hat = p.predict(x_hat)
        s = p.score(x, y)
        z = N - 1 if (d == 2) else 0
        label = u'%d阶，$R^2$=%.3f' % (d, s)
        plt.plot(x_hat, y_hat, color=clrs[i], lw=line_width[i], alpha=0.75,label=label, zorder=z)
        plt.legend(loc='upper left')
        plt.grid(True)
       # plt.title('线性回归', fontsize=18)
        plt.xlabel('X', fontsize=16)
        plt.ylabel('Y', fontsize=16)
    plt.show()
```
运行代码后可见打印控制台信息如下：
![20170428163609687](https://i.loli.net/2018/07/07/5b4071502ee4f.jpg)
图像如下：
![20170428163704786](https://i.loli.net/2018/07/07/5b407184ece95.jpg)
### 4.线性回归的推广(linear/Ridge/Lasso/ElasticNet)
从上面图像可以看出，当模型复杂度提高的时候，对训练集的数据拟合很好，但会出现过度拟合现象，为了防止这种过拟合现象的出现，我们在损失函数中加入了惩罚项，根据惩罚项不同分为以下：
![20170428165846484](https://i.loli.net/2018/07/07/5b4071ae1551e.jpg)
最后一个为Elastic Net 回归，把 L1 正则和 L2 正则按一定的比例结合起来

L1会趋向于产生少量的特征，而其他的特征都是0，而L2会选择更多的特征，这些特征都会接近于0。Lasso在特征选择时候非常有用，而Ridge就只是一种规则化而已。在所有特征中只有少数特征起重要作用的情况下，选择Lasso比较合适，因为它能自动选择特征。而如果所有特征中，大部分特征都能起作用，而且起的作用很平均，那么使用Ridge也许更合适。对于各种回归的比较可以看下图：
![20170428170254959](https://i.loli.net/2018/07/07/5b4071dc45cd3.jpg)
目前线性回归就写这么多，有时间再慢慢修改和添加更详细的内容。

##二、Logistic回归
在正式进入主题之前， 先提前强调一下，logistic回归，又叫对数几率回归（从后文中可以知道这个名字的由来），`这是一个分类模型而不是一个回归模型！`,下文开始将从不同方面讲解logistic回归的原理，随后分别使用梯度上升算法和随机梯度上升算法将logistic回归算法应用到实例中.
### 1.原理介绍
#### 1.1 前言
想必大家也早有疑惑，既然logistic回归名字中都带有“回归”二者，难道和回归模型一点关系都没有！`没错，二者是有联系的`，下面我们便来谈一谈.
由上面的讲解，我们已经知道线性回归的模型如下：
$$ f(x)=	\omega_0x_0+	\omega_1x_1 +...+	\omega_nx_n$$
写成向量形式：
$$ f(x) =	\omega^Tx+b $$
同时进一步可以写成`广义`线性回归模型：
$$ f(x) = g^{-1}(\omega^Tx+b)$$
(注意，其中$g^{-1}(x)$是单调可微函数。)
#### 1.2 logistic回归和线性回归的关系
`注意了，敲黑板了哈！！！
下面我们便从线性回归的回归模型引出logistic回归的分类模型！！！`
我们知道上诉线性回归模型只能够进行回归学习，但是若要是做分类任务如何做！答案便是在“广义线性回归”模型中：只需找一个单调可微函数将分类任务的真实标记y与线性回归模型的预测值联系起来便可以了！

logistic回归是处理二分类问题的，所以输出的标记y={0,1}，并且线性回归模型产生的预测值z=wx+b是一个实值，所以我们将实值z转化成0/1值便可，这样有一个可选函数便是“单位阶跃函数”：
$$ f(n)= \begin{cases}
 0, & \text {z<0} \\
 0.5, & \text{z=0}\\
 1,&\text{z>0}
 \end{cases} $$
 这种如果预测值大于0便判断为正例，小于0则判断为反例，等于0则可任意判断！
 但是单位阶跃函数是非连续的函数，我们需要一个连续的函数，“Sigmoid函数”便可以很好的取代单位阶跃函数：
 $$  y=\frac{1}{1+e^{-z}} \quad $$
 sigmoid函数在一定程度上近似单位阶跃函数，同时单调可微，图像如下所示：
 ![20170320171903777](https://i.loli.net/2018/07/07/5b40d899a62db.png)
 这样我们在原来的线性回归模型外套上sigmoid函数便形成了logistic回归模型的预测函数，可以用于二分类问题：
 $$ y =\frac{1}{1+e^{-(\omega^Tx+b)}} \quad$$
对上式的预测函数做一个变换：
$$ ln\frac{y}{1-y}\quad = \omega^Tx+b $$
观察上式可得：若将y视为样本x作为正例的可能性，则1-y便是其反例的可能性。二者的比值便被称为“几率”，反映了x作为正例的相对可能性，`这也是logistic回归又被称为对数几率回归的原因`！

`这里我们也便可以总结一下线性回归模型和logistic回归的关系： `
logistic回归分类模型的预测函数是在用线性回归模型的预测值的结果去逼近真实标记的对数几率！这样也便实现了上面说的将线性回归的预测值和分类任务的真实标记联系在了一起！
### 2.模型推导
#### 2.1 建模
在上一个话题中我们已经得到了logistic回归的预测函数：
$$ y =\frac{1}{1+e^{-(\omega^Tx+b)}} \quad \dots(1)$$
$$ ln\frac{y}{1-y}\quad = \omega^Tx+b \dots(2)$$
这里我们将式子中的y视为类后验概率估计p(y=1|x)，则上式可以重写为：
$$ ln\frac{p(y=1|x)}{p(y=0|x)}\quad = \omega^Tx+b $$
这儿有必要解释一下，其中：
$p(y=1|x)$：表示样本x被预测为正例的可能性大小
$p(y=0|x)$：表示样本x被预测为反例的可能性大小
则 $ln\frac{p(y=1|x)}{p(y=0|x)}\quad$：表示被预测为正例的相对可能性
根据$p(y=1|x)+p(y=0|x)=1$，解上面的方程有：
$$p(y=1|x)=\frac{e^{\omega^Tx+b}}{1+e^{\omega^Tx+b}}\quad=h_\omega(x)$$
$$p(y=0|x)=\frac{1}{1+e^{\omega^Tx+b}}\quad=1-h_\omega(x)$$
那下面剩下的任务就是求解参数$\omega$了，思路如下：
1、为求解参数w，我们需要定义一个准则函数 J(w)，利用准则函数求解参数w
2、我们通过最大似然估计法定义准则函数J(w)
3、接下来通过最大化准则函数J(w)便可求出参数w的迭代表达式
4、为了更好地使用数据求出参数w，我们将第三步得到的w的迭代时向量化。自此便完成了对于参数w的推导过程，接下来便可以进行实例应用了
#### 2.2 参数计算
根据上面的过程，我们知道了求解参数大致分为三步，我们一起来看看吧。。。。
`第一步：`求解准则函数$J(\omega)$
合并（3）（4）两个式子可得：
$$p(y|x,\omega)=(h_\omega(x))^y(1-h_\omega(x))^{1-y}  \dots(5)$$
在（5）式中y={0,1}，是个二分类问题，所以y只是取两个值0或是1。
根据（5）式可得似然函数为：
$$L(\omega)=\quad \prod_{i=1}^m p(y^i|x^i\omega)(1-h_\omega(x^i))^{1-y^i} \dots(6)$$
对（6）式取对数有：
$$l(\omega)=lnL(\omega)=\sum_1^m(y^ilnh_\omega(x^i)+(1-y^i)ln(1-h_w(x^i))) \dots(7)$$
因此定义准则函数为：
$$J(\omega)=\frac{1}{m}\quad l(\omega)\sum_1^m(y^ilnh_\omega(x^i)+(1-y^i)ln(1-h_w(x^i))) \dots(8)$$
所以，我们最终的目标就是最大化似然函数，即求解准则函数的最大值。
$$max_\omega J(\omega) \dots(9) $$

`第二步：`梯度上升算法求解参数$\omega$
这里我们使用梯度上升算法求解参数$\omega$，因此参数$\omega$的迭代式为：
$$\omega_{j+1}=\omega_j+\alpha \nabla J(\omega_j)$$
其中$\alpha$是正的比例因子，用于设定步长的“学习率”
其中对准则函数$J(w)$进行微分可得：
$$ \frac{\partial J(\omega_j)}{\partial \omega_j}=\nabla J(\omega_j)=\frac{1}{m}\sum_1^m(h_\omega(x^i-y^i)x_j^i)$$
所以得到最终参数w的迭代式为：
$$ \omega_{j+i}=w_j+\alpha \frac{1}{m}\sum_1^m(h_\omega(x^i-y^i)x_j^i)$$
上式将$\frac{1}{m}$去掉不影响结果，等价于下式：
$$ \omega_{j+i}=w_j+\alpha \sum_1^m(h_\omega(x^i-y^i)x_j^i)$$

至此我们已经得出了$\omega$的迭代公式，按说是可以在引入数据的情况下进行$\omega$的计算，进而进行分类！但是数据基本都是以矩阵和向量的形式引入的，所以我们需要对上面$\omega$的迭代时进行向量化，以方便实例应用中的使用。
`第三步：`$\omega$迭代公式向量化
首先对于引入的数据集$x$来说，均是以矩阵的形式引入的，如下：
$$
x=\begin{bmatrix}
      x^1 \\
      \vdots \\
      x^m \\
      \end{bmatrix}
    =\begin{bmatrix}
      x_0^1 \cdots x_n^1\\
      \vdots \ddots \vdots \\
      x_0^m  \cdots x_n^m \\
      \end{bmatrix}
$$
其中m数据的个数，n是数据的维度，也就是数据特征的数量！
再者便是标签y也是以向量的形式引入的：
$$
y=\begin{bmatrix}
      y^1 \\
      \vdots \\
      y^m \\
      \end{bmatrix}
$$
参数$\omega$向量化为：
$$
\omega =\begin{bmatrix}
      \omega_0 \\
      \vdots \\
      \omega_n \\
      \end{bmatrix}
$$

在这里定义$M=x*\omega$，所以：
$$
M=x\omega
  =\begin{bmatrix}
    \omega_0x_0^1+ \cdots +\omega_nx_n^1\\
    \vdots \ddots \vdots \\
    \omega_0x_0^m+  \cdots +\omega_nx_n^m \\
    \end{bmatrix}
$$
定义上面说的sigmoid函数为：
 $$ g(x) =\frac{1}{1+e^{-(\omega^Tx+b)}} \quad$$
所以定义估计的误差损失为：
$$
E=h_\omega(x)-y
 =\begin{bmatrix}
       g(M^1)-y^1 \\
       \vdots \\
       g(M^m)-y^m \\
       \end{bmatrix}
  =\begin{bmatrix}
        e^1 \\
        \vdots \\
        e^m\\
        \end{bmatrix}
  =g(A)-y
$$
在此基础上，可以得到步骤二中得到的参数迭代时向量化的式子为：
$$ \omega_{j+1}= \omega+\alpha x^TE$$
至此我们便完成了参数w迭代公式的推导，下面便可以在实例中应用此迭代公式进行实际的分析了！下面我们便以简单的二分类问题为例来分析logistic回归算法的使用！！

### 3.代码详情
此实例便是在二维空间中给出了两类数据点，现在需要找出两类数据的分类函数，并且对于训练出的新的模型，如果输入新的数据可以判断出该数据属于二维空间中两类数据中的哪一类！

在给出Python实现的示例代码展示之前，先介绍一下两种优化准则函数的方法：
#### 3.1 梯度上升算法：
梯度上升算法和我们平时用的梯度下降算法思想类似，梯度上升算法基于的思想是：要找到某个函数的最大值，最好的方法是沿着这个函数的梯度方向探寻！直到达到停止条件为止！
梯度上升算法的伪代码：
```
每个回归系数都初始化为1；
重复R次：
     计算整个数据集的梯度；
     使用（学习率X梯度）来更新系数的向量;
     返回回归系数
```
#### 3.2 随机梯度上升算法：
梯度上升算法在每次更新回归系数时都需要遍历整个数据集，该方法在处理小数据时还尚可，但如果有数十亿样本和成千上万的特征，那么该方法的计算复杂度太高了，改进方法便是一次仅用一个数据点来更新回归系数，此方法便称为随机梯度上升算法！由于可以在更新样本到来时对分类器进行增量式更新，因而随机梯度上升算法是一个“在线学习算法”。而梯度上升算法便是“批处理算法”！
**但是这儿出现了一个问题，随机梯度上升算法虽然大大减少了计算复杂度，但是同时正确率也下降了！所以可以对随机梯度上升算法进行改进！**
#### 3.3 改进的随机梯度上升算法：
进分为两个方面：
+ 改进一、对于学习率alpha采用非线性下降的方式使得每次都不一样
+ 改进二：每次使用一个数据，但是每次随机的选取数据，选过的不在进行选择
+ 改进三：也可以在每次迭代式进行小批量处理，也就是每次随机选取一定数量的样本（可以是10个，20个，100个。。。），这样就可以兼顾梯度上升和随机梯度上升的性能.

在知道这些信息之后下面给出示例代码，其中有详细的注释：
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numpy import *
import matplotlib.pyplot as plt

#从文件中加载数据：特征X，标签label
def loadDataSet():
    dataMatrix=[]
    dataLabel=[]
    #这里给出了python 中读取文件的简便方式
    f=open('testSet.txt')
    for line in f.readlines():
        #print(line)
        lineList=line.strip().split()
        dataMatrix.append([1,float(lineList[0]),float(lineList[1])])
        dataLabel.append(int(lineList[2]))
    #for i in range(len(dataMatrix)):
    #   print(dataMatrix[i])
    #print(dataLabel)
    #print(mat(dataLabel).transpose())
    matLabel=mat(dataLabel).transpose()
    return dataMatrix,matLabel

#logistic回归使用了sigmoid函数
def sigmoid(inX):
    return 1/(1+exp(-inX))

#函数中涉及如何将list转化成矩阵的操作：mat()
#同时还含有矩阵的转置操作：transpose()
#还有list和array的shape函数
#在处理矩阵乘法时，要注意的便是维数是否对应

#graAscent函数实现了梯度上升法，隐含了复杂的数学推理
#梯度上升算法，每次参数迭代时都需要遍历整个数据集
def graAscent(dataMatrix,matLabel):
    m,n=shape(dataMatrix)
    matMatrix=mat(dataMatrix)

    w=ones((n,1))
    alpha=0.001
    num=500
    for i in range(num):
        error=sigmoid(matMatrix*w)-matLabel
        w=w-alpha*matMatrix.transpose()*error
    return w


#随机梯度上升算法的实现，对于数据量较多的情况下计算量小，但分类效果差
#每次参数迭代时通过一个数据进行运算
def stocGraAscent(dataMatrix,matLabel):
    m,n=shape(dataMatrix)
    matMatrix=mat(dataMatrix)

    w=ones((n,1))
    alpha=0.001
    num=20  #这里的这个迭代次数对于分类效果影响很大，很小时分类效果很差
    for i in range(num):
        for j in range(m):
            error=sigmoid(matMatrix[j]*w)-matLabel[j]
            w=w-alpha*matMatrix[j].transpose()*error
    return w

#改进后的随机梯度上升算法
#从两个方面对随机梯度上升算法进行了改进,正确率确实提高了很多
#改进一：对于学习率alpha采用非线性下降的方式使得每次都不一样
#改进二：每次使用一个数据，但是每次随机的选取数据，选过的不在进行选择
def stocGraAscent1(dataMatrix,matLabel):
    m,n=shape(dataMatrix)
    matMatrix=mat(dataMatrix)

    w=ones((n,1))
    num=200  #这里的这个迭代次数对于分类效果影响很大，很小时分类效果很差
    setIndex=set([])
    for i in range(num):
        for j in range(m):
            alpha=4/(1+i+j)+0.01

            dataIndex=random.randint(0,100)
            while dataIndex in setIndex:
                setIndex.add(dataIndex)
                dataIndex=random.randint(0,100)
            error=sigmoid(matMatrix[dataIndex]*w)-matLabel[dataIndex]
            w=w-alpha*matMatrix[dataIndex].transpose()*error
    return w

#绘制图像
def draw(weight):
    x0List=[];y0List=[];
    x1List=[];y1List=[];
    f=open('testSet.txt','r')
    for line in f.readlines():
        lineList=line.strip().split()
        if lineList[2]=='0':
            x0List.append(float(lineList[0]))
            y0List.append(float(lineList[1]))
        else:
            x1List.append(float(lineList[0]))
            y1List.append(float(lineList[1]))

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(x0List,y0List,s=10,c='red')
    ax.scatter(x1List,y1List,s=10,c='green')

    xList=[];yList=[]
    x=arange(-3,3,0.1)
    for i in arange(len(x)):
        xList.append(x[i])

    y=(-weight[0]-weight[1]*x)/weight[2]
    for j in arange(y.shape[1]):
        yList.append(y[0,j])

    ax.plot(xList,yList)
    plt.xlabel('x1');plt.ylabel('x2')
    plt.show()


if __name__ == '__main__':
    dataMatrix,matLabel=loadDataSet()
    #weight=graAscent(dataMatrix,matLabel)
    weight=stocGraAscent1(dataMatrix,matLabel)
    print(weight)
    draw(weight)
```
[测试数据链接](https://pan.baidu.com/s/1qXWZQbU)(密码：1l1s)
上面程序运行结果为：
![Figure_1](https://i.loli.net/2018/07/08/5b418a7589649.png)
图（1）是使用梯度上升算法的结果，运算复杂度高；
![Figure_2![Uploading Figure_1-1.png… (lbijqgeam)]()](https://i.loli.net/2018/07/08/5b418ab32e55a.png)
图（2）是使用随机梯度上升算法，分类效果略差吗，运算复杂度低；
![Figure_1-1](https://i.loli.net/2018/07/08/5b418b17b5b12.png)
图（3）是使用改进后的随机梯度上升算法，分类效果好，运算复杂度低。

### 4.Logistic回归的推广
(暂无)
