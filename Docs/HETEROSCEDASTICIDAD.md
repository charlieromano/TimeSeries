# HETEROSCEDASTICIDAD

Características comunes de series de tiempo de finanzas, modelo ARCH, ejemplos.



**Ref: [Springer.TimeSeriesAnalysis2008]**

Sea {Y_t } una serie de tiempo de interés. La varianza condicional de Y_t mide la incerteza en el desvío de Y_t a partir de su media condicional.
$$
E(Y_t | Y_{t-1}, Y_{t-2}, ...)
$$
Si {Y_t } sigue un modelo ARIMA, la varianza condicional (a un paso) es siempre igual  a la varianza del ruido  para cada valor presente y pasado de cualquier número fijo de pasos adelante para un proceso ARIMA. 

En la práctica, la varianza condicional es un proceso aleatorio en si mismo. Por ejemplo los retornos diarios de una acción suelen tener valores grandes de varianza condicional cuando siguen un período de variaciones violentas de precio, en comparación con períodos más estables.

El objetivo central de este tema es modelar el proceso de varianza condicional para predecir la varianza de los valores futuros, basándonos en los valores del presente y del pasado. Cuando estudiamos ARIMA, estamos interesados en predecir el valor medio futuro basado en las muestras, por ejemplo el precio de una acción.  Ahora estamos interesados en predecir la varianza. 



* Retorno

$$
r_t = \log(p_t)-\log(p_{t-1})
$$



* Modelo ARCH(1)

