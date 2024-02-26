#โครงสร้าง 3 input nodes, 2 hidden nodes, 1 output node
#มี input ชุดเดียว คำนวณ backprop 1 ครั้งเพื่อปรับค่า
# import numpy 
from NNfunction import * # ถ้า import * คือเอาทุก function ในไฟล์ NNfunction
import pandas as pd
import random
from Test import *

learning_rate = 0.9
desire_output = [1, 1]  # 1,1 is 2 เป็นมะเร็ง 0,0 is 4 ไม่เป็น
error = 1

X = [5, 2, 1, 1, 2, 1.0, 3, 1, 1, 2]
Weight10 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
Weight11 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
Weight12 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
Weight13 = [0.1, 0.2, 1]
Weight14 = [0.1, 0.2, 1]

list_correct_train = []
list_correct_test = []

round_count =10
for j in range(round_count):
       #random weight 
    Weight10 = [random.uniform(-1,1) for i in range(10)]
    Weight11 = [random.uniform(-1,1) for i in range(10)]
    Weight12 = [random.uniform(-1,1) for i in range(10)]
    Weight13 = [random.uniform(-1,1) for i in range(4)] 
    Weight14 = [random.uniform(-1,1) for i in range(4)]

    avg_error=1
    count_epoch=0

    while count_epoch<10:
       avg_error = 0
       error = 0
       data = pd.read_csv("train_dataset.csv")

       for i in range(len(data)):
              X=data.iloc[i,0:10].tolist()
              desire_output = [1,1]if X[9] == 2 else [0,0]
              X[9]=1
              #forward pass
              print("\n-----Forward pass-----> ")

              out10=Nout(X,Weight10) 
              y10=sigmoid(out10) 
              print("\nSum(V) of node 10 is: %8.3f, Y from node 10is: %8.3f" % (out10,y10))

              out11=Nout(X,Weight11) 
              y11=sigmoid(out11) 
              print("\nSum(V) of node 11 is: %8.3f, Y from node 11 is: %8.3f" % (out11,y11))

              out12=Nout(X,Weight12) 
              y12=sigmoid(out12) 
              print("\nSum(V) of node 12 is: %8.3f, Y from node 12 is: %8.3f" % (out12,y12))


              out13=Nout([y10,y11,y12,1],Weight13) # from 10,11,12
              y13=sigmoid(out13) 
              print("\nSum(V) of node 13 is: %8.3f, Y from node 13 is: %8.3f" % (out13,y13))

              out14=Nout([y10,y11,y12,1],Weight14) # from 10,11,12
              y14=sigmoid(out14) 
              print("\nSum(V) of node 14 is: %8.3f, Y from node 14 is: %8.3f" % (out14,y14))

              #Error
              error13=desire_output[0]-y13
              error14=desire_output[1]-y14
              print("\nError of node 13 is: %8.3f, Error of node 14 is: %8.3f" % (error13, error14))
              avg_error = (error13 + error14) / 2


              #backpropagation
              print("\n-----backward pass-----> ")
              
              #node14
              grad14=gradOut(error14,y14) 
              delta_wieght_w10_14 = deltaw(learning_rate,grad14,y10)
              delta_wieght_w11_14 = deltaw(learning_rate,grad14,y11)
              delta_wieght_w12_14 = deltaw(learning_rate,grad14,y12)
              delta_bias14 = deltaw(learning_rate,grad14,1)
              Weight14 = [Weight14[0]+delta_wieght_w10_14 , Weight14[1]+delta_wieght_w11_14 , Weight14[2]+delta_wieght_w12_14 , delta_bias14]
              print("\nNew weights for node 14: weight10_14: %8.3f, weight11_14: %8.3f, weight12_14: %8.3f, Bias: %8.3f" % (Weight14[0], Weight14[1], Weight14[2], Weight14[3]))

              #node13
              grad13=gradOut(error13,y13) 
              delta_wieght_w10_13 = deltaw(learning_rate,grad13,y10)
              delta_wieght_w11_13 = deltaw(learning_rate,grad13,y11)
              delta_wieght_w12_13 = deltaw(learning_rate,grad13,y12)
              delta_bias13 = deltaw(learning_rate,grad13,1)
              Weight13 = [Weight13[0]+delta_wieght_w10_13 , Weight13[1]+delta_wieght_w11_13 , Weight13[2]+delta_wieght_w12_13 , delta_bias13]
              print("\nNew weights for node 13: weight10_13: %8.3f, weight11_13: %8.3f, weight12_13: %8.3f, Bias: %8.3f" % (Weight13[0], Weight13[1], Weight13[2], Weight13[3]))

              #node12
              sum_from_node_13_14 = (Weight14[2]*grad14)+(Weight13[2]*grad13)
              grad12=gradH(y12,sum_from_node_13_14)
              delta_wieght_w1_12 = deltaw(learning_rate,grad12,X[0])
              delta_wieght_w2_12 = deltaw(learning_rate,grad12,X[1])
              delta_wieght_w3_12 = deltaw(learning_rate,grad12,X[2])
              delta_wieght_w4_12 = deltaw(learning_rate,grad12,X[3])
              delta_wieght_w5_12 = deltaw(learning_rate,grad12,X[4])
              delta_wieght_w6_12 = deltaw(learning_rate,grad12,X[5])
              delta_wieght_w7_12 = deltaw(learning_rate,grad12,X[6])
              delta_wieght_w8_12 = deltaw(learning_rate,grad12,X[7])
              delta_wieght_w9_12 = deltaw(learning_rate,grad12,X[8])
              delta_bias12 = deltaw(learning_rate,grad12,1)
              Weight12 = [Weight12[0]+delta_wieght_w1_12 ,Weight12[1]+delta_wieght_w2_12 ,Weight12[2]+delta_wieght_w3_12 ,Weight12[3]+delta_wieght_w4_12 ,Weight12[4]+delta_wieght_w5_12 ,
                            Weight12[5]+delta_wieght_w6_12 ,Weight12[6]+delta_wieght_w7_12 ,Weight12[7]+delta_wieght_w8_12 ,Weight12[8]+delta_wieght_w9_12 , delta_bias12]
              print("\nNew weights for node 12: weight1_12: %8.3f, weight2_12: %8.3f, weight3_12: %8.3f, weight4_12: %8.3f, weight5_12: %8.3f, weight6_12: %8.3f, weight7_12: %8.3f, weight8_12: %8.3f, weight9_12: %8.3f, Bias: %8.3f"
                     % (Weight12[0], Weight12[1], Weight12[2], Weight12[3], Weight12[4], Weight12[5], Weight12[6], Weight12[7], Weight12[8], Weight12[9]))

              #node11
              sum_from_node_13_14 = (Weight14[1]*grad14)+(Weight13[1]*grad13)
              grad11=gradH(y11,sum_from_node_13_14)
              delta_wieght_w1_11 = deltaw(learning_rate,grad11,X[0])
              delta_wieght_w2_11 = deltaw(learning_rate,grad11,X[1])
              delta_wieght_w3_11 = deltaw(learning_rate,grad11,X[2])
              delta_wieght_w4_11 = deltaw(learning_rate,grad11,X[3])
              delta_wieght_w5_11 = deltaw(learning_rate,grad11,X[4])
              delta_wieght_w6_11 = deltaw(learning_rate,grad11,X[5])
              delta_wieght_w7_11 = deltaw(learning_rate,grad11,X[6])
              delta_wieght_w8_11 = deltaw(learning_rate,grad11,X[7])
              delta_wieght_w9_11 = deltaw(learning_rate,grad11,X[8])
              delta_bias11 = deltaw(learning_rate,grad11,1)
              Weight11 = [Weight11[0]+delta_wieght_w1_11 ,Weight11[1]+delta_wieght_w2_11 ,Weight11[2]+delta_wieght_w3_11 ,Weight11[3]+delta_wieght_w4_11 ,Weight11[4]+delta_wieght_w5_11 ,
                            Weight11[5]+delta_wieght_w6_11 ,Weight11[6]+delta_wieght_w7_11 ,Weight11[7]+delta_wieght_w8_11 ,Weight11[8]+delta_wieght_w9_11 , delta_bias11]
              print("\nNew weights for node 11: weight1_11: %8.3f, weight2_11: %8.3f, weight3_11: %8.3f, weight4_11: %8.3f, weight5_11: %8.3f, weight6_11: %8.3f, weight7_11: %8.3f, weight8_11: %8.3f, weight9_11: %8.3f, Bias: %8.3f"
                     % (Weight11[0], Weight11[1], Weight11[2], Weight11[3], Weight11[4], Weight11[5], Weight11[6], Weight11[7], Weight11[8], Weight11[9]))
              
              #node10
              sum_from_node_13_14 = (Weight14[1]*grad14)+(Weight13[1]*grad13)
              grad10=gradH(y10,sum_from_node_13_14)
              delta_wieght_w1_10 = deltaw(learning_rate,grad10,X[0])
              delta_wieght_w2_10 = deltaw(learning_rate,grad10,X[1])
              delta_wieght_w3_10 = deltaw(learning_rate,grad10,X[2])
              delta_wieght_w4_10 = deltaw(learning_rate,grad10,X[3])
              delta_wieght_w5_10 = deltaw(learning_rate,grad10,X[4])
              delta_wieght_w6_10 = deltaw(learning_rate,grad10,X[5])
              delta_wieght_w7_10 = deltaw(learning_rate,grad10,X[6])
              delta_wieght_w8_10 = deltaw(learning_rate,grad10,X[7])
              delta_wieght_w9_10 = deltaw(learning_rate,grad10,X[8])
              delta_bias10 = deltaw(learning_rate,grad10,1)
              Weight10 = [Weight10[0]+delta_wieght_w1_10 ,Weight10[1]+delta_wieght_w2_10 ,Weight10[2]+delta_wieght_w3_10 ,Weight10[3]+delta_wieght_w4_10 ,Weight10[4]+delta_wieght_w5_10 ,
                            Weight10[5]+delta_wieght_w6_10 ,Weight10[6]+delta_wieght_w7_10 ,Weight10[7]+delta_wieght_w8_10 ,Weight10[8]+delta_wieght_w9_10 , delta_bias10]
              print("\nNew weights for node 10: weight1_10: %8.3f, weight2_10: %8.3f, weight3_10: %8.3f, weight4_10: %8.3f, weight5_10: %8.3f, weight6_10: %8.3f, weight7_10: %8.3f, weight8_10: %8.3f, weight9_10: %8.3f, Bias: %8.3f"
                     % (Weight10[0], Weight10[1], Weight10[2], Weight10[3], Weight10[4], Weight10[5], Weight10[6], Weight10[7], Weight10[8], Weight10[9]))
              avg_error /= len(data)
              count_epoch+=1

       print("\n-------------------End of Training------------------->")
    print("\nAverage Error: ", avg_error)
    print("\nTotal Epochs: ", count_epoch)
    print("\nFinal weights of node 10 are: ", Weight10)
    print("\nFinal weights of node 11 are: ", Weight11)
    print("\nFinal weights of node 12 are: ", Weight12)
    print("\nFinal weights of node 13 are: ", Weight13)
    print("\nFinal weights of node 14 are: ", Weight14)


    list_correct_train.append(experimental(Weight10, Weight11, Weight12, Weight13, Weight14, count_epoch, avg_error, False))
    list_correct_test.append(experimental(Weight10, Weight11, Weight12, Weight13, Weight14, count_epoch, avg_error, True))




import matplotlib.pyplot as plt

avg_correct_train = sum(list_correct_train) / len(list_correct_train)
avg_correct_test = sum(list_correct_test) / len(list_correct_test)
x_values = list(range(1, round_count + 1))

plt.scatter(x_values, list_correct_train, label="Train")
plt.scatter(x_values, list_correct_test, label="Test")
plt.axhline(avg_correct_train, color="r", linestyle="--", label="Average Train")
plt.axhline(avg_correct_test, color="g", linestyle="-.", label="Average Test")
plt.title("Correct Prediction of Train and Test Data")
plt.xlabel("Trial No.")
plt.ylabel("Correct Prediction")
plt.legend()
plt.show()