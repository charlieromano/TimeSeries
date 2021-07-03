### AR models



AR(1) model:
$$
y_t = a_1 y_{t-1}+\epsilon_t
$$
AR(2) model:
$$
y_t = a_1 y_{t-1}+a_2 y_{t-2}+\epsilon_t
$$
AR(p) model:
$$
y_t = a_1 y_{t-1}+a_2 y_{t-2}+...+a_p y_{t-p}+\epsilon_t
$$

### MA models



MA(1) model:
$$
y_t = b_1\epsilon_{t-1}+\epsilon_t
$$
MA(2) model:
$$
y_t = b_1\epsilon_{t-1}+b_2\epsilon_{t-2}+\epsilon_t
$$
MA(q) model
$$
y_t = b_1\epsilon_{t-1}+b_2\epsilon_{t-2}+...+b_q\epsilon_{t-q}+\epsilon_t
$$

#### ARMA  models

AR+MA (1,1):
$$
y_t = a_1y_{t-1} + b_1\epsilon_{t-1}+\epsilon_t
$$
ARMA(p,q)
$$
y_t = a_1y_{t-1} + ... + a_py_{t-p} + b_1\epsilon_{t-1} +...+b_q\epsilon_{t-q} +\epsilon_t
$$

$$
y_t = \sum_{i=1}^pa_iy_{t-i} + \sum_{j=1}^q b_i\epsilon_{t-i} + \epsilon_t
$$

