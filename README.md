# Keyboard
键盘位置是固定的，是写死了，后续可以根据实际的情况进行修改。
每个key对应的正态分布，mean是对应的key的中点，然后协方差矩阵是选取的[[0.01,0.01],[0.01,0.01]],所以他的偏差也不大，基本集中在中点附近，可以之后进行修改。
选取mid-point是根据paper里的算法计算出来的
要import的库有matplotlib,numpy,scipy,math
