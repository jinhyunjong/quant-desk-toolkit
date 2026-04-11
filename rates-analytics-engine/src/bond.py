def bond_price(face, coupon_rate, maturity, yield_rate, freq=1):
    price = 0
    for t in range(1, maturity * freq + 1):
        price += (face * coupon_rate / freq) / ((1 + yield_rate / freq) ** t)
    price += face / ((1 + yield_rate / freq) ** (maturity * freq))
    return price
