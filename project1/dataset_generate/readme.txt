genrate.py用来生成一个具有四个特征的二分类数据集，用于预测某同学是否能够通过期末考试（Pass/Fail），结果将存储在
~/ml_hw1/dataset/grade_dataset.

特征：
（1）每天学习时长：lt: 1-8
（2）期中考试成绩：mid_grade:30-100
（3）出勤率：attendence_rate:30%-100%
（4）作业完成情况：homework 30%-100%

生成规则：
一般而言,每天学习时长lr与期末考试成绩相关程度最高，其次是期中考试成绩，作业完成情况和出勤率。我们将全部的数据都进行组内归一化，
并且默认按照以下公式来计算通过期末考试的概率：

p=0.5*lt+0.2*mid_grade+0.2*attendence_rate+0.1*homework

如果p大于0.5,我们认为该同学能够通过期末考试。上述公式将被用来生成linear_dataset。为了验证SVM在其他情况下的表现，我们还额外
考虑其他两个数据集:
noisy_dataset: 为阈值0.5引入±0.2的噪声
nonlinear_dataset:引入非线性映射，当0.25<p<0.75时认为能够通过期末考试。


