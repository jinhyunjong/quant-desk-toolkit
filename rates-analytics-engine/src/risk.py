def dv01(price_func, shift=0.0001):
    base = price_func(0)
    bumped = price_func(shift)
    return (bumped - base) / shift
