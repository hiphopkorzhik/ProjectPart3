

# функция задания начального условия
def u_init(x):
    return 11 - h * (x - 1)

# функция задания левого граничного условия
def u_left(x):
    return 11