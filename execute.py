# โครงสร้าง 3 input nodes, 2 hidden nodes, 1 output node
# มี input ชุดเดียว คำนวณ backprop 1 ครั้งเพื่อปรับค่า
# import numpy
from NNfunction import *  # ถ้า import * คือเอาทุก function ในไฟล์ NNfunction
import pandas as pd
import random
from Test import *



def traindataset(X, Weight10, Weight11, Weight12, Weight13, Weight14, desire_output, learning_rate):
    # forward pass
    print("\n-----Forward pass-----> ")

    out10 = Nout(X, Weight10)
    y10 = sigmoid(out10)
    print("\nSum(V) of node 10 is: %8.3f, Y from node 10is: %8.3f" % (out10, y10))

    out11 = Nout(X, Weight11)
    y11 = sigmoid(out11)
    print("\nSum(V) of node 11 is: %8.3f, Y from node 11 is: %8.3f" % (out11, y11))

    out12 = Nout(X, Weight12)
    y12 = sigmoid(out12)
    print("\nSum(V) of node 12 is: %8.3f, Y from node 12 is: %8.3f" % (out12, y12))

    out13 = Nout([y10, y11, y12, 1], Weight13)  # from 10,11,12
    y13 = sigmoid(out13)
    print("\nSum(V) of node 13 is: %8.3f, Y from node 13 is: %8.3f" % (out13, y13))

    out14 = Nout([y10, y11, y12, 1], Weight14)  # from 10,11,12
    y14 = sigmoid(out14)
    print("\nSum(V) of node 14 is: %8.3f, Y from node 14 is: %8.3f" % (out14, y14))

    # Error
    error13 = desire_output - y13
    error14 = desire_output - y14


    print("\nError of node 13 is: %8.3f, Error of node 14 is: %8.3f" % (error13, error14))
    error = (error13 + error14) / 2

    if error !=0 :
        # backpropagation
        print("\n-----backward pass-----> ")

        # node14
        grad14 = gradOut(error, y14)
        delta_wieght_w10_14 = deltaw(learning_rate, grad14, y10)
        delta_wieght_w11_14 = deltaw(learning_rate, grad14, y11)
        delta_wieght_w12_14 = deltaw(learning_rate, grad14, y12)
        delta_bias14 = deltaw(learning_rate, grad14, 1)
        Weight14 = [
            Weight14[0] + delta_wieght_w10_14,
            Weight14[1] + delta_wieght_w11_14,
            Weight14[2] + delta_wieght_w12_14,
            Weight14[3] + delta_bias14,
        ]
        print("\nNew weights for node 14: weight10_14: %8.3f, weight11_14: %8.3f, weight12_14: %8.3f, Bias: %8.3f"% (Weight14[0], Weight14[1], Weight14[2], Weight14[3]))

        # node13
        grad13 = gradOut(error, y13)
        delta_wieght_w10_13 = deltaw(learning_rate, grad13, y10)
        delta_wieght_w11_13 = deltaw(learning_rate, grad13, y11)
        delta_wieght_w12_13 = deltaw(learning_rate, grad13, y12)
        delta_bias13 = deltaw(learning_rate, grad13, 1)
        Weight13 = [
            Weight13[0] + delta_wieght_w10_13,
            Weight13[1] + delta_wieght_w11_13,
            Weight13[2] + delta_wieght_w12_13,
            Weight13[3] + delta_bias13,
        ]
        print("\nNew weights for node 13: weight10_13: %8.3f, weight11_13: %8.3f, weight12_13: %8.3f, Bias: %8.3f"% (Weight13[0], Weight13[1], Weight13[2], Weight13[3]))

        # node12
        sum_from_node_13_14 = (Weight14[2] * grad14) + (Weight13[2] * grad13)
        grad12 = gradH(y12, sum_from_node_13_14)
        delta_wieght_w1_12 = deltaw(learning_rate, grad12, X[0])
        delta_wieght_w2_12 = deltaw(learning_rate, grad12, X[1])
        delta_wieght_w3_12 = deltaw(learning_rate, grad12, X[2])
        delta_wieght_w4_12 = deltaw(learning_rate, grad12, X[3])
        delta_wieght_w5_12 = deltaw(learning_rate, grad12, X[4])
        delta_wieght_w6_12 = deltaw(learning_rate, grad12, X[5])
        delta_wieght_w7_12 = deltaw(learning_rate, grad12, X[6])
        delta_wieght_w8_12 = deltaw(learning_rate, grad12, X[7])
        delta_wieght_w9_12 = deltaw(learning_rate, grad12, X[8])
        delta_bias12 = deltaw(learning_rate, grad12, 1)
        Weight12 = [
            Weight12[0] + delta_wieght_w1_12,
            Weight12[1] + delta_wieght_w2_12,
            Weight12[2] + delta_wieght_w3_12,
            Weight12[3] + delta_wieght_w4_12,
            Weight12[4] + delta_wieght_w5_12,
            Weight12[5] + delta_wieght_w6_12,
            Weight12[6] + delta_wieght_w7_12,
            Weight12[7] + delta_wieght_w8_12,
            Weight12[8] + delta_wieght_w9_12,
            Weight12[9] + delta_bias12,
        ]
        print("\nNew weights for node 12: weight1_12: %8.3f, weight2_12: %8.3f, weight3_12: %8.3f, weight4_12: %8.3f, weight5_12: %8.3f, weight6_12: %8.3f, weight7_12: %8.3f, weight8_12: %8.3f, weight9_12: %8.3f, Bias: %8.3f"% (
                Weight12[0],
                Weight12[1],
                Weight12[2],
                Weight12[3],
                Weight12[4],
                Weight12[5],
                Weight12[6],
                Weight12[7],
                Weight12[8],
                Weight12[9],
            ))

        # node11
        sum_from_node_13_14 = (Weight14[1] * grad14) + (Weight13[1] * grad13)
        grad11 = gradH(y11, sum_from_node_13_14)
        delta_wieght_w1_11 = deltaw(learning_rate, grad11, X[0])
        delta_wieght_w2_11 = deltaw(learning_rate, grad11, X[1])
        delta_wieght_w3_11 = deltaw(learning_rate, grad11, X[2])
        delta_wieght_w4_11 = deltaw(learning_rate, grad11, X[3])
        delta_wieght_w5_11 = deltaw(learning_rate, grad11, X[4])
        delta_wieght_w6_11 = deltaw(learning_rate, grad11, X[5])
        delta_wieght_w7_11 = deltaw(learning_rate, grad11, X[6])
        delta_wieght_w8_11 = deltaw(learning_rate, grad11, X[7])
        delta_wieght_w9_11 = deltaw(learning_rate, grad11, X[8])
        delta_bias11 = deltaw(learning_rate, grad11, 1)
        Weight11 = [
            Weight11[0] + delta_wieght_w1_11,
            Weight11[1] + delta_wieght_w2_11,
            Weight11[2] + delta_wieght_w3_11,
            Weight11[3] + delta_wieght_w4_11,
            Weight11[4] + delta_wieght_w5_11,
            Weight11[5] + delta_wieght_w6_11,
            Weight11[6] + delta_wieght_w7_11,
            Weight11[7] + delta_wieght_w8_11,
            Weight11[8] + delta_wieght_w9_11,
            Weight11[9] + delta_bias11,
        ]
        print(
            "\nNew weights for node 11: weight1_11: %8.3f, weight2_11: %8.3f, weight3_11: %8.3f, weight4_11: %8.3f, weight5_11: %8.3f, weight6_11: %8.3f, weight7_11: %8.3f, weight8_11: %8.3f, weight9_11: %8.3f, Bias: %8.3f"
            % (
                Weight11[0],
                Weight11[1],
                Weight11[2],
                Weight11[3],
                Weight11[4],
                Weight11[5],
                Weight11[6],
                Weight11[7],
                Weight11[8],
                Weight11[9],
            ))

        # node10
        sum_from_node_13_14 = (Weight14[0] * grad14) + (Weight13[0] * grad13)
        grad10 = gradH(y10, sum_from_node_13_14)
        delta_wieght_w1_10 = deltaw(learning_rate, grad10, X[0])
        delta_wieght_w2_10 = deltaw(learning_rate, grad10, X[1])
        delta_wieght_w3_10 = deltaw(learning_rate, grad10, X[2])
        delta_wieght_w4_10 = deltaw(learning_rate, grad10, X[3])
        delta_wieght_w5_10 = deltaw(learning_rate, grad10, X[4])
        delta_wieght_w6_10 = deltaw(learning_rate, grad10, X[5])
        delta_wieght_w7_10 = deltaw(learning_rate, grad10, X[6])
        delta_wieght_w8_10 = deltaw(learning_rate, grad10, X[7])
        delta_wieght_w9_10 = deltaw(learning_rate, grad10, X[8])
        delta_bias10 = deltaw(learning_rate, grad10, 1)
        Weight10 = [
            Weight10[0] + delta_wieght_w1_10,
            Weight10[1] + delta_wieght_w2_10,
            Weight10[2] + delta_wieght_w3_10,
            Weight10[3] + delta_wieght_w4_10,
            Weight10[4] + delta_wieght_w5_10,
            Weight10[5] + delta_wieght_w6_10,
            Weight10[6] + delta_wieght_w7_10,
            Weight10[7] + delta_wieght_w8_10,
            Weight10[8] + delta_wieght_w9_10,
             Weight10[9] + delta_bias10,
        ]
        print(
            "\nNew weights for node 10: weight1_10: %8.3f, weight2_10: %8.3f, weight3_10: %8.3f, weight4_10: %8.3f, weight5_10: %8.3f, weight6_10: %8.3f, weight7_10: %8.3f, weight8_10: %8.3f, weight9_10: %8.3f, Bias: %8.3f"
            % (
                Weight10[0],
                Weight10[1],
                Weight10[2],
                Weight10[3],
                Weight10[4],
                Weight10[5],
                Weight10[6],
                Weight10[7],
                Weight10[8],
                Weight10[9],
            )
        )
    return error, Weight10, Weight11, Weight12, Weight13, Weight14


def training_path(round_count):
    avg_error = 0
    learning_rate = -0.9
    Weight10 = [random.uniform(-1, 1) for i in range(10)]
    Weight11 = [random.uniform(-1, 1) for i in range(10)]
    Weight12 = [random.uniform(-1, 1) for i in range(10)]
    Weight13 = [random.uniform(-1, 1) for i in range(4)]
    Weight14 = [random.uniform(-1, 1) for i in range(4)]

    dataset = pd.read_csv('train_dataset.csv')
    # result = pd.read_csv()
    for j in range(round_count):
        avg_error = 0
        print(f"Round {j + 1}")
        for i in range(len(dataset)):
            data = dataset.iloc[i, 0:10].tolist()
            desire_output = 1 if data[9] == 2 else 0
            data[9] = 1
            error, Weight10, Weight11, Weight12, Weight13, Weight14 = traindataset(
                data, Weight10, Weight11, Weight12, Weight13, Weight14, desire_output, learning_rate)
            avg_error += error
        avg_error /= len(dataset)

    print("\n-------------------End of Training------------------->")
    print("\nAverage Error: ", avg_error)
    print("\nTotal Round: ", round_count)
    print("\nFinal weights of node 10 are: ", Weight10)
    print("\nFinal weights of node 11 are: ", Weight11)
    print("\nFinal weights of node 12 are: ", Weight12)
    print("\nFinal weights of node 13 are: ", Weight13)
    print("\nFinal weights of node 14 are: ", Weight14)

    return Weight10, Weight11, Weight12, Weight13, Weight14, avg_error


def test_path(Weight10, Weight11, Weight12, Weight13, Weight14, round_count, avg_error):
    data = pd.read_csv("test_dataset.csv")
    total = 0
    correct = 0
    for i in range(len(data)):
        X = data.iloc[i, 0:10].tolist()
        correct_ans = X[9]
        X[9] = 1

        # Forward
        print("\n-----Forward pass-----> ")

        out10 = Nout(X, Weight10)
        y10 = sigmoid(out10)
        print("\nSum(V) of node 10 is: %8.3f, Y from node 10is: %8.3f" % (out10, y10))

        out11 = Nout(X, Weight11)
        y11 = sigmoid(out11)
        print("\nSum(V) of node 11 is: %8.3f, Y from node 11 is: %8.3f" % (out11, y11))

        out12 = Nout(X, Weight12)
        y12 = sigmoid(out12)
        print("\nSum(V) of node 12 is: %8.3f, Y from node 12 is: %8.3f" % (out12, y12))

        out13 = Nout([y10, y11, y12, 1], Weight13)  # from 10,11,12
        y13 = sigmoid(out13)
        print("\nSum(V) of node 13 is: %8.3f, Y from node 13 is: %8.3f" % (out13, y13))

        out14 = Nout([y10, y11, y12, 1], Weight14)  # from 10,11,12
        y14 = sigmoid(out14)
        print("\nSum(V) of node 14 is: %8.3f, Y from node 14 is: %8.3f" % (out14, y14))

        y13 = 1 if y13 > 0.5 else 0
        y14 = 1 if y14 > 0.5 else 0
        result = [y13, y14]
        result = 2 if result == [1, 1] else 4
        print("\nDesire Output: ", [1, 1] if X[9] == 2 else [0, 0])
        print("\nOutput: ", [y13, y14])
        print("Correct Answer: ", correct_ans, "Predicted Answer: ", result)
        print("Correct" if correct_ans == result else "Wrong")
        if correct_ans == result:
            correct += 1
        total += 1
        print("\n------------------------------------------------->")

    print("\n-------------------FINAL RESULT------------------->")
    print(
        "This test with model trained with ",
        round_count,
        " epochs with avg error: ",
        avg_error,
    )
    print("Total: ", total)
    print("Correct: ", correct)
    print("Wrong: ", total - correct)
    print("Accuracy: ", correct / total * 100, "%")
    return correct / total * 100


def calculate(round_count):
    Weight10, Weight11, Weight12, Weight13, Weight14, avg_error = training_path(round_count)
    return test_path(
        Weight10, Weight11, Weight12, Weight13, Weight14, round_count, avg_error
    )
