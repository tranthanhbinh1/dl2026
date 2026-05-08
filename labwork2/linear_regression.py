from pathlib import Path
from matplotlib import pyplot as plt


def read_csv(file_name: str):
    x_feature = []
    y_target = []

    with open(file_name, "r") as file:
        for line in file:
            x, y = line.split(",")
            x_feature.append(float(x.strip()))
            y_target.append(float(y.strip()))

    return x_feature, y_target


def predict(x: float, w: float, b: float):
    return w * x + b


def cost_function(x_feature, y_target, w: float, b: float):
    m = len(x_feature)
    total_error = 0

    for x, y in zip(x_feature, y_target):
        error = predict(x, w, b) - y
        total_error += error**2

    return total_error / (2 * m)


def gradient_descent(
    x_feature, y_target, w: float, b: float, learning_rate: float, epochs: int
):
    m = len(x_feature)

    for i in range(epochs):
        dj_dw = 0
        dj_db = 0

        for x, y in zip(x_feature, y_target):
            error = predict(x, w, b) - y
            dj_dw += error * x
            dj_db += error

        dj_dw = dj_dw / m
        dj_db = dj_db / m

        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db

        if i % 1000 == 0:
            cost = cost_function(x_feature, y_target, w, b)
            print(f"epoch: {i}, w: {w}, b: {b}, cost: {cost}")

    return w, b


def plot_regression_line(x_feature, y_target, w: float, b: float):
    predicted_y = []

    for x in x_feature:
        predicted_y.append(predict(x, w, b))

    plt.scatter(x_feature, y_target, color="blue", label="Actual data")
    plt.plot(x_feature, predicted_y, color="red", label="Regression line")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression using Gradient Descent")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data_file = Path(__file__).with_name("lr.csv")
    x_feature, y_target = read_csv(data_file)

    w = 0
    b = 0
    learning_rate = 0.0001
    epochs = 10000

    w, b = gradient_descent(x_feature, y_target, w, b, learning_rate, epochs)

    print(f"Final model: y = {w}x + {b}")
    print(f"Prediction for x=50: {predict(50, w, b)}")

    plot_regression_line(x_feature, y_target, w, b)
