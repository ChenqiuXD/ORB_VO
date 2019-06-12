# 解释

pp: position & posture。在代码中`pp`具有以下的形式：

```python
pp = np.array([<x>, <y>, <z>, <theta_x>, <theta_y>, <theta_z>])
```

# Ransac及相关函数

*Random Sample Consensus*

相关函数均在`class ORBDetector`内部，有：

1. `def rotate_matrix(axis, radian)`：*static*。利用旋转轴和弧度计算，使用矩阵的指数方法来计算（`scipy.linalg.expm`函数）。原本用于函数`def getT(pp, three_d=False)`和`ransac_residual_func(x, cord_list=None, is_lm=False, three_d=False)`，但**因为计算效率较低，目前已经不再使用**，留在类里面作为备用。

2. `def getT(pp, three_d=False)`：*static*。根据布尔参数`three_d`来决定采用3D旋转矩阵或2D旋转矩阵。三维情形返回的矩阵为当前坐标系沿着自身先平移后沿着平移后的自身旋转的转移矩阵T。

3. `def ransac_residual_func(x, cord_list=None, is_lm=False, three_d=False)`：*static*。Ransac使用的计算误差的函数，可以用于`scipy.optimize.least_squares()`（`is_lm=True`），也可以用来计算某个`pp`在`cord_list`上的误差（`is_lm=False`）。同`getT`，本函数中`three_d`可以表示3D或者2D的计算。

4. `def optimize_ransac(self, three_d=False)`：*非static*。采用Ransac过程，嵌入LM Least Squares算法进行优化找到最优`pp`。

   Ransac过程流程：

   1. 使用`self.camera_coordinate_first`和`self.camera_coordinate_second`进行初始化工作。
   2. 设定最大迭代次数、每次抽样数、阈值、break阈值等参数。
   3. 随机抽取6对Match（Maybe inliers），使用least\_squares计算这6个的`pp`。
   4. 在未被抽取的点中逐一使用3中计算出的`pp`计算误差，如果小于阈值则认为是Also inliers。
   5. 如果Also inliers+Maybe inliers的数量超过阈值，根据Also inliers + Maybe inliers重新计算`pp`，返回，结束。否则进入6
   6. 计算3中`pp`对所有的`cord_list`的误差，然后与`best_err`比较，如果小则更新`best_pp`。
   7. 判断是否超过最大循环次数，如果超过则结束，未超过则返回3。

   时间上需要优化。

   