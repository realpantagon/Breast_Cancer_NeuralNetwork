
from NNfunction import *


def experimental(w10, w11, w12, w13,w14, count_epoch, avg_error, is_train=False):
    if is_train:
        data = pd.read_csv("train_dataset.csv")
    else:
        data = pd.read_csv("test_dataset.csv")
    total = 0
    correct = 0
    for i in range(len(data)):
        X = data.iloc[i, 0:10].tolist()
        correct_ans = X[9]
        X[9] = 1
        # Forward
        # print("\n-------------------Forward------------------->")
        out10 = Nout(X, w10)
        y10 = sigmoid(out10)

        out11 = Nout(X, w11)
        y11 = sigmoid(out11)

        out12 = Nout(X, w12)
        y12 = sigmoid(out12)

        out13 = Nout([y10,y11,y12, 1], w13)
        y13 = sigmoid(out13)

        out14 = Nout([y10,y11,y12, 1], w14)
        y14 = sigmoid(out14)

        y13 = 1 if y13 > 0.5 else 0
        y14 = 1 if y14 > 0.5 else 0
        result = [y13, y14]
        result = 2 if result == [1, 1] else 4
        # print("\nDesire Output: ", [1, 1] if X[9] == 2 else [0, 0])
        # print("\nOutput: ", [y12, y13])
        # print("Correct Answer: ", correct_ans, "Predicted Answer: ", result)
        # print("Correct" if correct_ans == result else "Wrong")
        if correct_ans == result:
            correct += 1
        total += 1
        # print("\n------------------------------------------------->")

    print("\n-------------------FINAL RESULT------------------->")
    print("Train data" if is_train else "Test data")
    print(" epochs with average error: ", avg_error)
    print("round_count : ", total)
    print("Correct: ", correct)
    print("Wrong: ", total - correct)
    print("Accuracy: ", correct / total * 100, "%")
    return correct / total * 100
