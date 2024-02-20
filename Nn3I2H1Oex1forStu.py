#โครงสร้าง 3 input nodes, 2 hidden nodes, 1 output node
#มี input ชุดเดียว คำนวณ backprop 1 ครั้งเพื่อปรับค่า
# import numpy 
from NNfunction import * # ถ้า import * คือเอาทุก function ในไฟล์ NNfunction

X=([1,0,1,1])
W4=([0.2,0.4,-0.5,-0.4]) #weight ที่เกี่ยวข้องกับ node 4 [w14,w24,w34,bias4]
W5=([-0.3,0.1,0.2,0.2]) #weight ที่เกี่ยวข้องกับ node 5 [w15,w25,w35,bias5]
W6 = [-0.3, -0.2, 0.1]
d6=1 
l=-0.9 
e6=1
count=1
while(e6>=0.1) and (count<=1000):
    #forward pass
    print("\n-----Forward pass-----> ")
    o4=Nout(X,W4) #call NNfunction
    y4=sigmoid(o4) #call NNfunction
    print("\nSum(V) of node 4 is: %8.3f, Y from node 4 is: %8.3f" % (o4,y4))

    o5 = Nout(X, W5) # Calculate the sum of inputs for node 5
    y5 = sigmoid(o5) # Calculate the output of node 5 using sigmoid function
    print("\nSum(V) of node 5 is: %8.3f, Y from node 5 is: %8.3f" % (o5, y5))


    X6 = [y4, y5, 1] # The inputs for node 6 are the outputs of nodes 4 and 5, plus the bias
    o6 = Nout(X6, W6)
    y6 = sigmoid(o6)
    print("\nSum(V) of node 6 is: %8.3f, Y from node 6 is: %8.3f" % (o6, y6))

    # Forward pass for node 6


    #backpropagation
    #node 6
    print("\n<---- Back propagation & calculate new Weights and Biases ----")
    e6=d6-y6
    g6=gradOut(e6,y6) #call NNfunction
    dw46=deltaw(l,g6,y4) #call NNfunction
    w46n=W6[0]+dw46
    dw56=deltaw(l,g6,y5)
    w56n=W6[1]+dw56
    db6=deltaw(l,g6,1)
    b6n=W6[2]+db6
    W6 = np.array([w46n, w56n, b6n])

    # print("\nNew w46 is %8.3f, New w56 is:%8.3f, New bias 6 is %8.3f"% (w46n,w56n,b6n))
    print("\nNew weights for node 6: w46: %8.3f, w56: %8.3f, Bias: %8.3f" % (w46n, w56n, b6n))


    #node5
    # pre gradient5=g6*w56
    # Calculate the gradient for node 5 (backpropagation from node 6)
    pre_grad5 = g6 * W6[1]  # The part of the gradient from node 6 to node 5
    g5 = gradH(y5, pre_grad5)  # Gradient for node 5

    # Update weights for node 5
    dw15 = deltaw(l, g5, X[0])
    dw25 = deltaw(l, g5, X[1])
    dw35 = deltaw(l, g5, X[2])
    db5 = deltaw(l, g5, 1)  # Bias update

    # Calculate new weights for node 5
    w15n = W5[0] + dw15
    w25n = W5[1] + dw25
    w35n = W5[2] + dw35
    b5n = W5[3] + db5
    W5 = np.array([w15n, w25n, w35n, b5n])

    print("\nNew weights for node 5: w15: %8.3f, w25: %8.3f, w35: %8.3f, Bias: %8.3f" % (w15n, w25n, w35n, b5n))


    #node4
    #pre gradient4=g6*w46
    pre_grad4 = g6 * W6[0]  # The part of the gradient from node 6 to node 4
    g4 = gradH(y4, pre_grad4)  # Gradient for node 4

    # Update weights for node 4
    dw14 = deltaw(l, g4, X[0])
    dw24 = deltaw(l, g4, X[1])
    dw34 = deltaw(l, g4, X[2])
    db4 = deltaw(l, g4, 1)  # Bias update

    # Calculate new weights for node 4
    w14n = W4[0] + dw14
    w24n = W4[1] + dw24
    w34n = W4[2] + dw34
    b4n = W4[3] + db4
    W4 = np.array([w14n, w24n, w34n, b4n])

    print("\nNew weights for node 4: w14: %8.3f, w24: %8.3f, w34: %8.3f, Bias: %8.3f" % (w14n, w24n, w34n, b4n))
    print("Round = " ,count)
    count+=1