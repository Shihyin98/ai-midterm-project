# Code
> 程式碼

## [cnn-mnist.py](cnn-mnist.py)
* 利用**卷積神經網路模型(CNN Model)**，做資料集(datasets)的 **MNIST手寫數字識別**。

### Output
```
[1, 100] loss: 2.305
[1, 200] loss: 2.301
[1, 300] loss: 2.299
[1, 400] loss: 2.297
[1, 500] loss: 2.295
[1, 600] loss: 2.291
[1, 700] loss: 2.286
[1, 800] loss: 2.276
[1, 900] loss: 2.260
第1个epoch的識別準確率為：42%
[2, 100] loss: 2.188
[2, 200] loss: 1.978
[2, 300] loss: 1.312
[2, 400] loss: 0.764
[2, 500] loss: 0.577
[2, 600] loss: 0.489
[2, 700] loss: 0.420
[2, 800] loss: 0.376
[2, 900] loss: 0.387
第2个epoch的識別準確率為：90%
[3, 100] loss: 0.320
[3, 200] loss: 0.337
[3, 300] loss: 0.292
[3, 400] loss: 0.273
[3, 500] loss: 0.280
[3, 600] loss: 0.253
[3, 700] loss: 0.254
[3, 800] loss: 0.236
[3, 900] loss: 0.238
第3个epoch的識別準確率為：94%
[4, 100] loss: 0.214
[4, 200] loss: 0.207
[4, 300] loss: 0.190
[4, 400] loss: 0.199
[4, 500] loss: 0.189
[4, 600] loss: 0.166
[4, 700] loss: 0.170
[4, 800] loss: 0.157
[4, 900] loss: 0.150
第4个epoch的識別準確率為：95%
[5, 100] loss: 0.137
[5, 200] loss: 0.154
[5, 300] loss: 0.138
[5, 400] loss: 0.146
[5, 500] loss: 0.130
[5, 600] loss: 0.141
[5, 700] loss: 0.130
[5, 800] loss: 0.128
[5, 900] loss: 0.125
第5个epoch的識別準確率為：96%
[6, 100] loss: 0.129
[6, 200] loss: 0.114
[6, 300] loss: 0.105
[6, 400] loss: 0.112
[6, 500] loss: 0.114
[6, 600] loss: 0.103
[6, 700] loss: 0.114
[6, 800] loss: 0.118
[6, 900] loss: 0.113
第6个epoch的識別準確率為：97%
[7, 100] loss: 0.118
[7, 200] loss: 0.096
[7, 300] loss: 0.099
[7, 400] loss: 0.095
[7, 500] loss: 0.099
[7, 600] loss: 0.095
[7, 700] loss: 0.100
[7, 800] loss: 0.095
[7, 900] loss: 0.086
第7个epoch的識別準確率為：97%
[8, 100] loss: 0.090
[8, 200] loss: 0.085
[8, 300] loss: 0.088
[8, 400] loss: 0.100
[8, 500] loss: 0.087
[8, 600] loss: 0.090
[8, 700] loss: 0.077
[8, 800] loss: 0.080
[8, 900] loss: 0.093
第8个epoch的識別準確率為：97%
```

## [cnn-exercise.py](cnn-exercise.py)

### CNN Model
```
CNN_Model(
  (cnn1): Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1))
  (relu1): ReLU()
  (maxpool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (cnn2): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1))
  (relu2): ReLU()
  (maxpool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=512, out_features=10, bias=True)
)
```

### Output
```
Train Epoch: 1/20 Traing_Loss: 0.022266140207648277 Traing_acc: 95.966667% Val_Loss: 0.015899410471320152 Val_accuracy: 97.941666%
Train Epoch: 2/20 Traing_Loss: 0.003589425003156066 Traing_acc: 98.224998% Val_Loss: 0.09887941181659698 Val_accuracy: 97.974998%
Train Epoch: 3/20 Traing_Loss: 0.00350749958306551 Traing_acc: 98.537498% Val_Loss: 0.05649518966674805 Val_accuracy: 98.391670%
Train Epoch: 4/20 Traing_Loss: 0.010404400527477264 Traing_acc: 98.727081% Val_Loss: 0.051312245428562164 Val_accuracy: 98.608330%
Train Epoch: 5/20 Traing_Loss: 0.020619221031665802 Traing_acc: 98.891670% Val_Loss: 0.027847884222865105 Val_accuracy: 98.349998%
Train Epoch: 6/20 Traing_Loss: 0.06311257928609848 Traing_acc: 98.991669% Val_Loss: 0.0037723970599472523 Val_accuracy: 98.466667%
Train Epoch: 7/20 Traing_Loss: 0.18420739471912384 Traing_acc: 98.872917% Val_Loss: 0.00016567230341024697 Val_accuracy: 98.133331%
Train Epoch: 8/20 Traing_Loss: 0.24791915714740753 Traing_acc: 98.960419% Val_Loss: 0.04526807367801666 Val_accuracy: 98.349998%
Train Epoch: 9/20 Traing_Loss: 0.17582498490810394 Traing_acc: 99.118752% Val_Loss: 0.08589600771665573 Val_accuracy: 98.416664%
Train Epoch: 10/20 Traing_Loss: 0.0028617428615689278 Traing_acc: 99.270836% Val_Loss: 0.07057323306798935 Val_accuracy: 98.516670%
Train Epoch: 11/20 Traing_Loss: 0.1206415444612503 Traing_acc: 99.210419% Val_Loss: 0.1562708467245102 Val_accuracy: 98.400002%
Train Epoch: 12/20 Traing_Loss: 0.026923824101686478 Traing_acc: 99.020836% Val_Loss: 0.026145409792661667 Val_accuracy: 98.400002%
Train Epoch: 13/20 Traing_Loss: 0.006851387210190296 Traing_acc: 99.152084% Val_Loss: 0.00910208746790886 Val_accuracy: 98.233330%
Train Epoch: 14/20 Traing_Loss: 1.3856887562724296e-05 Traing_acc: 99.331253% Val_Loss: 0.11647797375917435 Val_accuracy: 98.500000%
Train Epoch: 15/20 Traing_Loss: 0.004589533898979425 Traing_acc: 99.425003% Val_Loss: 0.15453936159610748 Val_accuracy: 98.341667%
Train Epoch: 16/20 Traing_Loss: 0.09698867052793503 Traing_acc: 99.300003% Val_Loss: 0.1332489401102066 Val_accuracy: 97.908333%
Train Epoch: 17/20 Traing_Loss: 0.09293471276760101 Traing_acc: 99.297920% Val_Loss: 0.0977931022644043 Val_accuracy: 98.466667%
Train Epoch: 18/20 Traing_Loss: 0.003092012368142605 Traing_acc: 99.349998% Val_Loss: 0.2083318531513214 Val_accuracy: 98.349998%
Train Epoch: 19/20 Traing_Loss: 7.143020411604084e-06 Traing_acc: 99.356247% Val_Loss: 0.3610977828502655 Val_accuracy: 98.400002%
Train Epoch: 20/20 Traing_Loss: 8.115768650895916e-06 Traing_acc: 99.414581% Val_Loss: 0.1801408976316452 Val_accuracy: 98.458336%
```
    



# Note
* CNN 解決的問題所具備的**三個性質**：
  * 1. 局部性
    對於一張圖片而言，需要檢測圖片中的**特徵來決定圖片的類別**，通常情況下這些特徵都不是由整張圖片決定的，而是由一些**局部的區域決定**的。
    Ex: 在某張圖片中的某個局部檢測出了鳥喙，那麼基本可以判定圖片中有鳥這種動物。
  * 2. 相同性
    對於不同的圖片，它們具有同樣的特徵，這些特徵會出現在圖片的不同位置，也就是說可以用同樣的檢測模式去**檢測不同圖片的相同特徵**。
    Ex: 例如在不同的圖片中，雖然鳥喙處於不同的位置，但是我們可以用相同的模式去檢測。
  * 3. 不變性
    對於一張圖片，如果我們進行下採樣，那麼**圖片的性質基本保持不變**。

