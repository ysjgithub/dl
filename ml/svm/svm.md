 ## 什么是支持向量机？

 支持向量机又叫最大间隔分类器，假设现在只把目光放在二分类的任务上，对于线性可分的一组数据，找到一条直线把他们分开，这就完成了二分类任务，为了使这个模型对于新来的数据的泛化误差达到最小，当然希望把这批数据分的越开越好。最好的结果当然就是找到一条距离两类数据都是最远的一条直线
 
 - 找到距离这条线（超平面最近的点）
 - 最近的点的距离最大

 这些最近的点被称为支持向量，由于支持向量对于这个超平面的位置影响最大，所以被称为支持向量机
 
 ### 求解支持向量机

 一个超平面可表示为 $w^Tx+b=0$，对于一个超平面来说，w 和b 就是这个超平面的参数
 
 由二维的点到直线的距离公式，可扩展到N维的点到平面的距离公式 $r=\frac{\mid w^T+b\mid}{\mid\mid w\mid\mid}$ 
 
 $\begin{cases}
 w^Tx_i+b\geq+1,\quad y_i=+1\\
 w^Tx_i+b\leq-1,\quad y_i=-1
 \end{cases}$

可以看出$y_i(w^Tx_i+b)\geq0$，因此$d=y_i(w^Tx_i+b)\frac{1}{||w||}$，d的大小表示离分割超平面的距离的远近，可得到目标函数：找到一组$w$和$b$使得支持向量距离超平面的距离最大。

$\arg\max\limits_{w,b}{\min \limits_n(y_i(w^Tx_i+b)\frac{1}{||w||})}$

先找到支持向量，再优化$w$和$b$，使得距离最大。由于等比例扩大缩小$w$和$b$超平面的位置不会变化，我们就可以等比例缩放参数，使得目标函数更加友好

函数间隔：$\hat{\gamma_i}=y_i(w^Tx_i+b)$

几何间隔：$\gamma_i=\frac{y_i(w^Tx_i+b)}{||w||}$

函数间隔相当于参数乘上$||w||$

目标函数变为

$\arg\max\limits_{w,b}\frac{\hat{\gamma_i}}{||w||}$

$s.t.\quad y_i(w^Tx_i+b)\geq1 \qquad i=1,2,3...k$

若 $w$ 和 $b$ 同时除以 $\hat{\gamma}$ ,则 $\hat{\gamma}=\min y_i(w^Tx_i+b)=1$

 目标函数变为
 $\arg \max \limits_{w,b}\frac{1}{||w||}$
 
 $s.t.\quad y_i(w^Tx_i+b)\geq1, \qquad i=1,2,3...k$

将最大化问题转化为求最小值

$\min \limits_{w,b}\frac{1}{2}||w||^2$

$s.t.\quad y_i(w^Tx_i+b)\geq1, \qquad i=1,2,3...k$
  
这便是一个线性不等式约束下的二次优化问题, 下面我本就使用拉格朗日乘子法来获取我们优化目标的对偶形式。




### 拉格朗日乘子法
设求$\min \limits_{x}f(x) \quad s.t.\quad h(x)=0$

拉格朗日乘子法就是引入拉格朗日乘子，将约束加入目标函数
原式转化为$\min F(x,\lambda)=f(x)+\lambda h(x)$
通过求解$\nabla F(x,\lambda)=0,h(x)=0$即可求得最小值

### KKT条件
KKT条件是将约束为不等约束时如何转化的问题
设求$\min \limits_{x}f(x) \quad s.t.\quad h_1(x)\leq0,h_2(x)\leq0$

对应的拉格朗日函数为$\min F(x,\lambda_1,\lambda_2)=f(x)+\lambda_1 h_1(x)+\lambda_2 h_2(x)$
条件约束为
 $\begin{cases}
 h_1,h_2\geq0\\
 \nabla F(x,\lambda_1,\lambda_2)=0\\
 \lambda_i h_i(x)=0 \quad i = 1,2
 \end{cases}$

图片可以看出
case1:f(x)最优点不在h(x)的范围内，那么约束下的最优点应该在$h(x)=0$的直线上

case2:f(x)最优点在h(x)的范围内，那么对应的h(x)约束即没有起到作用对应的$\lambda=0$。假设在最优点 $x^*$ , $\nabla f(x^*) =-\lambda\nabla h(x^*)$,此时$h(x^*),f(x^*)$的梯度方向相反，因为我们希望得到$\min f(x)$，因此使$\lambda\geq0$

### 回到svm
目标函数变为$L(w,b,\lambda)=\frac{1}{2}||w||^2+\sum \limits_{i=1}^N\lambda(1-y_i(w^Tx_i+b))$

为了求解$\min L(w,b,\lambda)$,先将$\lambda$看作常数，求解$\min L(w,b)$的极值点
 $\begin{cases}
 \nabla_wL(w,b)=w-\sum \limits_{i=1}^N\lambda_ix_iy_i=0 \Rightarrow w =\sum \limits_{i=1}^N\lambda_ix_iy_i\\
 \nabla_bL(w,b)=-\sum \limits_{i=1}^N\lambda_iy_i=0 
 \end{cases}$
 
 代回$L(w,b,\lambda)$ 得到$L(w,b,\lambda)=W(\lambda)=\sum \limits_{i=1}^N\lambda_i-\frac{1}{2}\sum\limits_{i=1}^N\sum\limits_{j=1}^Ny_iy_j \lambda_i \lambda_j<x_i,x_j>$

问题转化为
$arg \max \limits_{\lambda} \sum \limits_{i=1}^N \lambda_i -\frac{1}{2} \sum \limits_{i=1}^N \sum \limits_{j=1}^N 
y_iy_j\lambda_i\lambda_jX_iX_j \quad s.t.\quad \lambda_i\geq0,\quad \sum \limits_{i=1}^{N}\lambda_iy_i=0$

### SMO算法

假设选取两个需要优化的参数$\lambda_1,\lambda_2$,剩下的$\lambda_3,\lambda_4...\lambda_N$固定，作为常数$constant$，则
$W(\lambda_1,\lambda_2)=\lambda_1+\lambda_2-\frac{1}{2}K_{1,1}\lambda_1^2y_1^2-\frac{1}{2}K_{2,2}\lambda_2^2y_2^2-K_{1,2}\lambda_1\lambda_2y_1y_2-y_1\lambda_1\sum\limits_{i=3}^N\lambda_iy_iK_{1,i}-y_2\lambda_2\sum\limits_{i=3}^N\lambda_iy_iK_{2,i}+C$
由约束条件$\sum \limits_{i=1}^{N}\lambda_iy_i=0$可得 $\lambda_1y_1+\lambda_2y_2=-\sum\limits_{i=3}^N\lambda_iy_i=\theta,\quad \lambda_1y_1=\theta-\lambda_2y_2$

左右同乘 $y_1$ ,则 $\lambda_1=y_1\theta-\lambda_2y_1y_2$ 代入得
$W(\lambda_2)=\lambda_1+\lambda_2-\frac{1}{2}K_{1,1}(\theta-\lambda_2y_2)^2-\frac{1}{2}K_{2,2}\lambda_2^2y_2^2-K_{1,2}\lambda_2y_2(\theta-\lambda_2y_2)-(\theta-\lambda_2y_2)\sum\limits_{i=3}^N\lambda_iy_iK_{1,i}-y_2\lambda_2\sum\limits_{i=3}^N\lambda_iy_iK_{2,i}+constant$

$v_i=\sum\limits_{j=3}^N\lambda_jy_jK_{i,j}=f(x_i)-\sum\limits_{j=1}^2y_i\lambda_jK_{i,j}-b \quad y_i->{1,-1}$

$W(\lambda_2)=\lambda_1+\lambda_2-\frac{1}{2}K_{1,1}(\theta-\lambda_2y_2)^2-\frac{1}{2}K_{2,2}\lambda_2^2-K_{1,2}\lambda_2y_2(\theta-\lambda_2y_2)-(\theta-\lambda_2y_2)v_1-y_2\lambda_2v_2+constant$

$\nabla_{\lambda_2}=-y_1y_2+1-\frac{1}{2}K_{1,1}*2*(\lambda_2y_2-\theta)*y_2-K_{2,2}\lambda_2-K_{1,2}y_2(\theta-\lambda_2y_2-y_2\lambda_2)+v_1y_2-v_2y_2$

$\nabla_{\lambda_2}=-y_1y_2+1-K_{1,1}\lambda_2+K_{1,1}y_2\theta-K_{2,2}\lambda_2-K_{1,2}y_2\theta+K_{1,2}\lambda_2+K_{1,2}\lambda_2+v_1y_2-v_2y_2$

$\nabla_{\lambda_2}=-y_1y_2+1-\lambda_2(K_{1,1}-K_{2,2}+2K_{1,2})+K_{1,1}y_2\theta-K_{1,2}y_2\theta+v_1y_2-v_2y_2=0$

$\lambda_2(K_{1,1}-K_{2,2}+2K_{1,2})=-y_1y_2+1+K_{1,1}y_2\theta-K_{1,2}y_2\theta+v_1y_2-v_2y_2$

$\lambda_2^{new} =\frac{-y_1y_2+1+K_{1,1}y_2\theta-K_{1,2}y_2\theta+v_1y_2-v_2y_2}{K_{1,1}-K_{2,2}+2K_{1,2}}$将$v_1,v_2$代入得

$v_1-v_2=f(x_1)-f(x_2)-\lambda_2y_2(K_{1,1}+2K_{1,2}+K_{2,2})-\theta(K_{1,1}-K_{1,2})$此时得$v_1,v_2,\lambda_1,\lambda_2$都是旧的数据，$v_1-v_2$代入 $\lambda_2^{new}$ 得
$\lambda_2^{new} =\frac{-y_1y_2+1+\lambda_2^{old}(K_{1,1}-K_{2,2}+2K_{1,2})+y_2[f(x_1)-f(x_2)]}{K_{1,1}-K_{2,2}+2K_{1,2}}=\frac{\lambda_2^{old}(K_{1,1}-K_{2,2}+2K_{1,2})+y_2[f(x_1)-y_1]-y_2[f(x_2)-y_2]}{K_{1,1}-K_{2,2}+2K_{1,2}}$

$E_i=f(x_i)-y_i$  设 $\beta=K_{1,1}-K_{2,2}+2K_{1,2}$

$\lambda_2^{new} =\frac{\lambda_2^{old}\beta+y_2E_1-y2E_2}{\beta}=\lambda_2^{old}+\frac{y_2E_1-y2E_2}{\beta}$

### 修剪
回顾$\lambda$的约束
$\lambda_1y_1+\lambda_2y_2=-\sum\limits_{i=3}^N\lambda_iy_i=\theta,\quad C\geq\lambda\geq0, \quad k$代表软间隔常数

$\begin{cases}
 \lambda_1-\lambda_2=k\qquad  y_1=y_2\quad L=max(0,\lambda_1-\lambda_2)\quad R=min(C,C-\lambda_1+\lambda_2)\\
 \lambda_1+\lambda_2=k\qquad y_1\ne y_2\quad L=max(0,\lambda_1+\lambda_2-C)\quad R=min(C,\lambda_1+\lambda_2)
 \end{cases}$
 
 $C\geq\lambda_1,\lambda_2\geq 0$,

根据求得得$\lambda_2^{new}$和修剪规则对$\lambda_2^{new,unclip}$进行修剪

$\lambda_2^{cliped}=\begin{cases}
 H\qquad\lambda_2^{new,unclip}\geq H\\
 \lambda_2^{new,unclip}\qquad H\geq\lambda_2^{new,unclip}\geq L\\
 L\qquad  L\geq\lambda_2^{new,unclip}
 \end{cases}$

 得到了$\lambda_2$更新后的值，可以使用$\theta$不变得性质来得到$\lambda_1^{new}=\lambda_1^{old}+y_1y_2(\lambda_2^{old}-\lambda_2^{new})$

 ### 更新b的值
 当$\lambda$的值更新后，如果不再边界上，则说明smo算法选取的点是支持向量，则$y(w^Tx+b)=1$因此$w^Tx+b=y$,则$b=y-w^T$此时使用更新后的$\lambda$计算，因为只更新了一对$\lambda$,另一总算法
 $b=-E+\lambda_1^{old}y_1K_{1,1}+\lambda_1^{old}y_2k_{2,1}+b^{old}-\lambda_1^{new}y_1K_{1,1}-\lambda_1^{new}y_2k_{2,1}$





