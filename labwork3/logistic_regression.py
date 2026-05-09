from pathlib import Path
import math
import csv
from matplotlib import pyplot as plt


def read_csv(file_name: str):
    x_features = []
    y_target = []

    with open(file_name, "r") as file:
        reader = csv.reader(file)
        next(reader)

        for experience, salary, loan in reader:
            x_features.append([float(experience), float(salary)])
            y_target.append(float(loan))

    return x_features, y_target


def predict(x_features, weights, b: float):
    z = 0

    for x, w in zip(x_features, weights):
        z += w * x

    z += b
    return sigmoid(z)


# TODO: implement sigmoid
def sigmoid(x):
    exp_neg_x = math.exp(-x)
    denominator = 1 + exp_neg_x
    result = 1 / denominator

    return result


def cost_function(x_features, y_target, weights, b: float):
    m = len(x_features)
    total_cost = 0

    for x, y in zip(x_features, y_target):
        prediction = predict(x, weights, b)
        total_cost += -y * math.log(prediction) - (1 - y) * math.log(1 - prediction)

    return total_cost / m


def gradient_descent(
    x_features, y_target, weights, b: float, learning_rate: float, epochs: int
):
    m = len(x_features)

    for i in range(epochs):
        dj_dw = [0 for _ in weights]
        dj_db = 0

        for x, y in zip(x_features, y_target):
            error = predict(x, weights, b) - y

            for j in range(len(weights)):
                dj_dw[j] += error * x[j]

            dj_db += error

        dj_db = dj_db / m

        for j in range(len(weights)):
            dj_dw[j] = dj_dw[j] / m
            weights[j] = weights[j] - learning_rate * dj_dw[j]

        b = b - learning_rate * dj_db

        if i % 1000 == 0:
            cost = cost_function(x_features, y_target, weights, b)
            print(f"epoch: {i}, weights: {weights}, b: {b}, cost: {cost}")

    return weights, b


def plot_predictions(x_features, y_target, weights, b: float):
    if plt is None:
        print("Skipping plot because matplotlib is not installed.")
        return

    experience = []
    salary = []
    colors = []

    for x, y in zip(x_features, y_target):
        experience.append(x[0])
        salary.append(x[1])
        prediction = predict(x, weights, b)
        colors.append("red" if prediction >= 0.5 else "blue")

    plt.scatter(experience, salary, c=colors)

    if weights[1] != 0:
        min_experience = min(experience)
        max_experience = max(experience)
        line_x = [min_experience, max_experience]
        line_y = []

        for x in line_x:
            y = -(weights[0] * x + b) / weights[1]
            line_y.append(y)

        plt.plot(line_x, line_y, color="green", label="Decision boundary")

    plt.xlabel("Experience")
    plt.ylabel("Salary")
    plt.title("Loan Prediction using Logistic Regression")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data_file = Path(__file__).with_name("loan2.csv")
    x_features, y_target = read_csv(data_file)

    weights = [0, 0]
    b = 0
    learning_rate = 0.0001
    epochs = 10000

    weights, b = gradient_descent(
        x_features, y_target, weights, b, learning_rate, epochs
    )

    print(
        f"Final model: probability = sigmoid({weights[0]} * experience + {weights[1]} * salary + {b})"
    )
    print(f"Prediction for experience=2, salary=5: {predict([2, 5], weights, b)}")

    plot_predictions(x_features, y_target, weights, b)
