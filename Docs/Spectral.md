Introducción al Análisis Espectral



Análisis en el dominio del tiempo



Consideremos el proceso dado por la expresión:
$$
Y_t = R\cos(2\pi ft+ \Phi)
$$
donde $\Phi$ es una variable aleatoria, R la amplitud, f es la frecuencia y t el tiempo.

* amplitud
* frecuencia
* fase

La expresión anterior puede ser difíil de estimar ya que los parámetros R y $\Phi$ no son lineales. 

Usando trigonometría se puede transformar la expresión en:
$$
Y_t = R\cos(2\pi f t + \Phi) = A\cos(2\pi f t ) + B\sin(2\pi f t)
$$
y entonces los parámetros nuevos resultan de:
$$
\begin{align}
& A = R\cos(\Phi) & &B=-R\sin(\Phi)\\
& R =\sqrt{A^2+B^2}& & \Phi = \arctan(-B/A)
\end{align}
$$
luego se puede ajustar A y B para una frecuencia fija f a partir de los datos usando regresión por cuadrados mínimos.



Una combinación lineal general de senos y cosenos se puede escribir como:
$$
Y_t = A_0 + \sum_{j=1}^m (A_j\cos(2\pi f_j t) + B_j \sin(2\pi f_j t ))
$$
Con cuadrados mínimos ordinarios se puede ajustar los valores de los $A_j$ y $B_j$.

Fourier

Suponiendo que n es impar, esto es n=2k+1, se tiene que las frecuencias de la forma 1/n, 2/n, ..., k/n se denominan *frecuencias de Fourier*. Las variables predictoras del seno y coseno a tales frecuencias son ortogonales, y la estimación por cuadrados mínimos resulta simplemente:
$$
\begin{align}
&\hat{A}_0 = \bar{Y}\\
&\hat{A}_j = \frac{2}{n}\sum_{t=1}^n Y_t\cos(2\pi t j/n) && \hat{B}_j =  \frac{2}{n}\sum_{t=1}^nY_t\sin(2\pi t j /n)
\end{align}
$$


El resultado de la serie de suma de cosenos y senos es general. Esto es, cualquier serie de cualquier longitud n, sea determinística o estocástica y con o sin periodicidades verdaderas puede ajustarse poerfectamente por este modelo eligiendo m=n/2 si n es par y m=(n-1)/2 si n es impar. Luego habrá n parámetros a estimar para ajustar la serie de largo n.



Periodograma

Para muestras de tamaño impar, el periodogram se define como:
$$
I(j/n) = \frac{n}{2}(\hat{A}_j^2+\hat{B}_j^2)
$$
donde j/n es la frecuencia para j=1, 2, ..., k.

Si la muestra tiene tamaño par y n=2k, en la frecuencia de corte f=k/n=1/2 resulta:
$$
I(1/2) = n\hat{A}_k^2
$$
Como el periodograma es proporcional a la suma de los cuadrados de los coeficientes de regresión asociados con la frecuencia f=j/n, la altura del periodograma muestra la fuerza relativa de los pares seno y coseno en varias frecuencias del comportamiento general de la serie. En términos de estadística podemos usar el análisis de la varianza. El periodograma es la suma de los cuadrados con dos grados de libertad asociados con el par de coeficientes $(A_j,B_j)$  de la frecuencia j/n cuando n es impar: 
$$
\sum_{j=1}^n (Y_j-\hat{Y})^2 = \sum_{j=1}^k I(j/n)
$$
Como conclusión, en cualquiera de los casos se tiene que para series largas habrá que estimar un número grande de parámetros y para esto se usa la transformada rápida de Fourier, o FFT.



Análisis en el dominio de la frecuencia (análisis espectral)