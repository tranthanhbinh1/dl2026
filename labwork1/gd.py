def f(x):
    return x * x


def f_(x):
    return 2 * x


def gd(x: float, lr: float, epoch: int):
    for _ in range(epoch):
        x = x - lr * f_(x)
        print(f"{x} - {f(x)}")


if __name__ == "__main__":
    x = 10
    r = 0.1
    gd(x, r, 4)
