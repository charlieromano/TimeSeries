### ARMA (p,q)

$$
\phi(B)Y_t=\theta(B)e_t
$$

### ARIMA(p,d,q)

$$
\phi(B)(1-B)^dY_t=\theta(B)e_t
$$



### SARMA(p,q)(P,Q)s

$$
\phi(B)\Phi(B^s)Y_t=\theta(B)\Theta(B^s)e_t
$$



### SARIMA(p,d,q)(P,D,Q)s

$$
\phi(B)\Phi(B^s)((1-B)^d(1-B^s)^DY_t)=\theta(B)\Theta(B^s)e_t
$$





SAR(p)(P)s como AR(p) multiplicativo con AR(P)s del polinomio de la componente estacional 
$$
\phi(B)\Phi(B^s)Y_t=e_t
$$
SMA(q)(Q)s como MA(q) multiplicativo con MA(Q)s del polinomio de la componente estacional
$$
\theta(B)\Theta(B^s)e_t=Y_t
$$


ARIMA como ARMA diferenciado

Si
$$
W_t=\nabla^dY_t
$$
es ARMA, entonces Y_t es ARIMA.



SARIMA como SARMA diferenciado

Si
$$
W_t=\nabla^d\nabla_s^DY_t
$$
es SARMA, entonces Y_t es SARIMA



**Ejemplo** **SARIMA**: serie con periodo estacional 12 meses:
$$
\phi(B)\Phi(B^{12})((1-B)^d(1-B^{12})^DY_t)=\theta(B)\Theta(B^{12})e_t
$$
