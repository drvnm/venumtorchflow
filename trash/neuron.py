# forwaard pass
x = [1.0, -2.0, 3.0]
w = [-3.0, -1.0, 2.0]
b = 1

# inputs x respective weights
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]


# adding weighted inputs and bias
z = xw0 + xw1 + xw2 + b

# applying relu
y = max(0, z)
print(y)
dvalue = 1.0
drelu_dz = dvalue * (1. if z > 0 else 0.)

dsum_dxw0 = 1
drelu_dxw0 = drelu_dz * dsum_dxw0

dsum_dxw1 = 1
drelu_dxw1 = drelu_dz * dsum_dxw1

dsum_dxw2 = 1
drelu_dxw2 = drelu_dz * dsum_dxw2

dsum_db = 1
drelu_db = drelu_dz * dsum_db

dmul_dx0 = w[0]
dmul_dx1 = w[1]
dmul_dx2 = w[2]
dmul_dw0 = x[0]
dmul_dw1 = x[1]
dmul_dw2 = x[2]
drelu_dx0 = drelu_dxw0 * dmul_dx0
drelu_dw0 = drelu_dxw0 * dmul_dw0
drelu_dx1 = drelu_dxw1 * dmul_dx1
drelu_dw1 = drelu_dxw1 * dmul_dw1
drelu_dx2 = drelu_dxw2 * dmul_dx2
drelu_dw2 = drelu_dxw2 * dmul_dw2
drelu_db = drelu_dz * dsum_db
# print(drelu_dx0, drelu_dw0, drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2)


dx = [drelu_dx0, drelu_dx1, drelu_dx2]  # gradients on inputs
dw = [drelu_dw0, drelu_dw1, drelu_dw2]  # gradients on weights
db = drelu_db  # gradient on bias...just 1 bias here


print(w, b)
w[0] += -0.001 * dw[0]
w[1] += -0.001 * dw[1]
w[2] += -0.001 * dw[2]
b += -0.001 * db
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]


# adding weighted inputs and bias
z = xw0 + xw1 + xw2 + b

# applying relu
y = max(0, z)

print(y)



for i in range(444):
    dvalue = 1.0
    drelu_dz = dvalue * (1. if z > 0 else 0.)

    dsum_dxw0 = 1
    drelu_dxw0 = drelu_dz * dsum_dxw0

    dsum_dxw1 = 1
    drelu_dxw1 = drelu_dz * dsum_dxw1

    dsum_dxw2 = 1
    drelu_dxw2 = drelu_dz * dsum_dxw2

    dsum_db = 1
    drelu_db = drelu_dz * dsum_db

    dmul_dx0 = w[0]
    dmul_dx1 = w[1]
    dmul_dx2 = w[2]
    dmul_dw0 = x[0]
    dmul_dw1 = x[1]
    dmul_dw2 = x[2]
    drelu_dx0 = drelu_dxw0 * dmul_dx0
    drelu_dw0 = drelu_dxw0 * dmul_dw0
    drelu_dx1 = drelu_dxw1 * dmul_dx1
    drelu_dw1 = drelu_dxw1 * dmul_dw1
    drelu_dx2 = drelu_dxw2 * dmul_dx2
    drelu_dw2 = drelu_dxw2 * dmul_dw2
    drelu_db = drelu_dz * dsum_db
    # print(drelu_dx0, drelu_dw0, drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2)


    dx = [drelu_dx0, drelu_dx1, drelu_dx2]  # gradients on inputs
    dw = [drelu_dw0, drelu_dw1, drelu_dw2]  # gradients on weights
    db = drelu_db  # gradient on bias...just 1 bias here


  
    w[0] += -0.001 * dw[0]
    w[1] += -0.001 * dw[1]
    w[2] += -0.001 * dw[2]
    b += -0.001 * db
    xw0 = x[0] * w[0]
    xw1 = x[1] * w[1]
    xw2 = x[2] * w[2]


    # adding weighted inputs and bias
    z = xw0 + xw1 + xw2 + b

    # applying relu
    y = max(0, z)
    print(y)