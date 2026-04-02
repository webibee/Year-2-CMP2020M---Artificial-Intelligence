import math
import matplotlib.pyplot as plt

def plotLearningCurve(logged_error, graphname):
    x_data = []
    y_data = []
    x_data.extend([logged_error[i][0] for i in range(0, len(logged_error))])
    y_data.extend([logged_error[i][1] for i in range(0, len(logged_error))])
    fig, ax = plt.subplots()
    plt.title(graphname)
    ax.set(xlabel='Epoch', ylabel='Squared Error')
    ax.plot(x_data, y_data, 'tab:green')
    plt.show()
    
def calc_net (inputs, weights):
    net = 0
    for i in range(len(inputs)):
        net += inputs[i] * weights[i]
    return net

def function_calculation(net):
    return (1 / (1 + ((math.e)**(-1*net))))

inputs_and_outputs = [[1,0.5,1.00,0.75,1,0],[1,1.00,0.50,0.75,1,0],[1,1.00,1.00,1.00,1,0],[1,-0.01,0.50,0.25,0,1],[1,0.50,-0.25,0.13,0,1],[1,0.01,0.02,0.05,0,1]]

neuron_4_weights = [0.90, 0.74, 0.8, 0.35] #w04, w14, w24, w34
neuron_5_weights = [0.45, 0.13, 0.40, 0.97] #w05, w15, w25, w35
neuron_6_weights = [0.36, 0.68, 0.10, 0.96] #w06, w16, w26, w36
neuron_7_weights = [0.98, 0.35, 0.50, 0.90] #w07, w47, w57, w67
neuron_8_weights = [0.92, 0.80, 0.13, 0.80] #w08, w48, w58, w68   
neuron_7_output = 0
neuron_8_output = 0

logged_error = []

def Backpropagation(neuron_456_inputs, steps):
    current_lowest_epoch = 0
    current_lowest_epoch_error = 10
    for i in range(steps):
        sum_of_epoch_squared_errors = 0
        print("\n\n::::::::::::::::::::::::::::::::: STEP ", i+1, "::::::::::::::::::::::::::::::::::::::")
        for j in neuron_456_inputs:
            current_inputs = j[0:4] # x0, x1, x2, x3
            print("\n\nneurons_456_inputs::: ", current_inputs)
            neuron_7_desired_output = j[4]
            print("neuron_7_desired_output::: ", neuron_7_desired_output)
            neuron_8_desired_output = j[5]
            print("neuron_8_desired_output::: ", neuron_8_desired_output)
            
            #Forward Step
            neuron_4_output = function_calculation(calc_net(current_inputs, neuron_4_weights))
            neuron_5_output = function_calculation(calc_net(current_inputs, neuron_5_weights))
            neuron_6_output = function_calculation(calc_net(current_inputs, neuron_6_weights))

            neuron_78_inputs = [j[0], neuron_4_output, neuron_5_output, neuron_6_output] # j[0] == bias  
            neuron_7_output = calc_net(neuron_78_inputs, neuron_7_weights)
            neuron_8_output = calc_net(neuron_78_inputs, neuron_8_weights)

            #Backward Step
            #Output Errors
            output_7_error = neuron_7_desired_output - neuron_7_output
            output_8_error = neuron_8_desired_output - neuron_8_output

            #Hidden Errors
            hidden_4_error = neuron_4_output * (1 - neuron_4_output) * ((neuron_7_weights[1] * output_7_error)+(neuron_8_weights[1] * output_8_error))
            hidden_5_error = neuron_5_output * (1 - neuron_5_output) * ((neuron_7_weights[2] * output_7_error)+(neuron_8_weights[2] * output_8_error))
            hidden_6_error = neuron_6_output * (1 - neuron_6_output) * ((neuron_7_weights[3] * output_7_error)+(neuron_8_weights[3] * output_8_error))

            #weight updates
            learning_rate = 0.1
            update04 = learning_rate * hidden_4_error * current_inputs[0]
            update14 = learning_rate * hidden_4_error * current_inputs[1]
            update24 = learning_rate * hidden_4_error * current_inputs[2]
            update34 = learning_rate * hidden_4_error * current_inputs[3]
            update05 = learning_rate * hidden_5_error * current_inputs[0]
            update15 = learning_rate * hidden_5_error * current_inputs[1]
            update25 = learning_rate * hidden_5_error * current_inputs[2]
            update35 = learning_rate * hidden_5_error * current_inputs[3]
            update06 = learning_rate * hidden_6_error * current_inputs[0]
            update16 = learning_rate * hidden_6_error * current_inputs[1]
            update26 = learning_rate * hidden_6_error * current_inputs[2]
            update36 = learning_rate * hidden_6_error * current_inputs[3]
            
            update07 = learning_rate * output_7_error * current_inputs[0]
            update47 = learning_rate * output_7_error * neuron_4_output
            update57 = learning_rate * output_7_error * neuron_5_output
            update67 = learning_rate * output_7_error * neuron_6_output
            update08 = learning_rate * output_8_error * current_inputs[0]
            update48 = learning_rate * output_8_error * neuron_4_output
            update58 = learning_rate * output_8_error * neuron_5_output
            update68 = learning_rate * output_8_error * neuron_6_output

            #update changes
            neuron_4_weights[0] += update04
            print("W04: ", neuron_4_weights[0])
            neuron_4_weights[1] += update14
            print("W14: ", neuron_4_weights[1])
            neuron_4_weights[2] += update24
            print("W24: ", neuron_4_weights[2])
            neuron_4_weights[3] += update34
            print("W34: ", neuron_4_weights[3])
            neuron_5_weights[0] += update05
            print("W05: ", neuron_5_weights[0])
            neuron_5_weights[1] += update15
            print("W15: ", neuron_5_weights[1])
            neuron_5_weights[2] += update25
            print("W25: ", neuron_5_weights[2])
            neuron_5_weights[3] += update35
            print("W35: ", neuron_5_weights[3])
            neuron_6_weights[0] += update06
            print("W06: ", neuron_6_weights[0])
            neuron_6_weights[1] += update16
            print("W16: ", neuron_6_weights[1])
            neuron_6_weights[2] += update26
            print("W26: ", neuron_6_weights[2])
            neuron_6_weights[3] += update36
            print("W36: ", neuron_6_weights[3])
            neuron_7_weights[0] += update07
            print("W07: ", neuron_7_weights[0])
            neuron_7_weights[1] += update47
            print("W47: ", neuron_7_weights[1])
            neuron_7_weights[2] += update57
            print("W57: ", neuron_7_weights[2])
            neuron_7_weights[3] += update67
            print("W67: ", neuron_7_weights[3])
            neuron_8_weights[0] += update08
            print("W08: ", neuron_8_weights[0])
            neuron_8_weights[1] += update48
            print("W48: ", neuron_8_weights[1])
            neuron_8_weights[2] += update58
            print("W58: ", neuron_8_weights[2])
            neuron_8_weights[3] += update68
            print("W68: ", neuron_8_weights[3])
            
            sum_of_epoch_squared_errors += ((output_7_error)**2) + ((output_8_error)**2)
    
            #Softmax Function
            print("\nProbability Distribution Neuron 7::: ", ((math.e)**(neuron_7_output)) / (((math.e)**(neuron_7_output)) + ((math.e)**(neuron_8_output))))
            print("Probability Distribution Neuron 8::: ", ((math.e)**(neuron_8_output)) / (((math.e)**(neuron_7_output)) + ((math.e)**(neuron_8_output))))
            
        logged_error.append([i, ((sum_of_epoch_squared_errors)/12)])
        if sum_of_epoch_squared_errors/12 < current_lowest_epoch_error:
            current_lowest_epoch_error = sum_of_epoch_squared_errors/12
            current_lowest_epoch = i+1
            
    print("current_lowest_epoch:: ", current_lowest_epoch)
    print("current_lowest_epoch_error:: ", current_lowest_epoch_error)

Backpropagation(inputs_and_outputs, 10) #Training wih 1000 epochs, Testing with 100 epochs
plotLearningCurve(logged_error, "Learning Curve")
Backpropagation([[1, 0.3, 0.7, 0.9, 1, 0]], 1)