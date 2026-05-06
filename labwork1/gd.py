def f(x):
    return x * x


def f_(x):
    return 2 * x


def gd(x: float, lr: float, thres: float):
    x = x - lr * f_(x)
    while x >= thres:
        x = x - lr * f_(x)
        print(f"x: {x} - f(x): {f(x)}")
    print("Finished")


if __name__ == "__main__":
    x = 10
    r = 0.1
    gd(x, r, 0.1)
