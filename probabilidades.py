import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.stats import norm


#probabilidade de duas pessoas fazerem aniversario no mesmo dia em um grupo de p pessoas BINOMIAL
def probabilidade_binomial(p):
    P = 1 - (1-1/365)**(p*(p-1)/2)
    return P
# Cria valores de p de 1 a 720
p_values = np.arange(0, 120)
# Calcula os valores de probabilidade usando a funçao definida
probabilities = [probabilidade_binomial(p) for p in p_values]
# Plota o grafico
plt.plot(p_values, probabilities)
plt.title('Probabilidade binomial')
plt.xlabel('Número de Pessoas (p)')
plt.ylabel('Probabilidade')
plt.grid(True)
plt.show()





#probabilidade de duas pessoas fazerem aniversario no mesmo dia em um grupo de p pessoas POISSON
def probabilidade_poisson(p):
    P = 1 - math.exp(-p*(p-1)/730)
    return P
# Cria valores de p de 0 a 720
p_values = np.arange(0,120)
# Calcula os valores de probabilidade usando a funçao definida
probabilities = [probabilidade_poisson(p) for p in p_values]
# Plota o grafico
plt.plot(p_values, probabilities)
plt.title('Probabilidade Poisson')
plt.xlabel('Número de Pessoas (p)')
plt.ylabel('Probabilidade')
plt.grid(True)
plt.show()





#probabilidade m questoes de 15 com 4 alternativas cada
def acerto(m):
  comb=math.factorial(15)/(math.factorial(m)*math.factorial(15-m))
  P = comb*((1/4)**m)*(1-1/4)**(15-m)
  return P
print(acerto(3))
   #calculando os valores de m
m_values = np.arange(0, 16)  # Valores de m de 0 a 15
probabilities = [acerto(m) for m in m_values]

# Plotando o histograma
plt.figure(figsize=(8, 6))
plt.bar(m_values, probabilities)
plt.title('Histograma da Probabilidade de Acerto para m Questões')
plt.xlabel('Número de Questões Corretas (m)')
plt.ylabel('Probabilidade')
plt.grid(True)
plt.show()





#Em um grupo de pessoas, a media de altura foi 170 cm e o desvio padrao foi 5 cm
media = 170  # em cm
desvio_padrao = 5  # em cm
# Calculando a altura acima da qual estão os 10% mais altos
altura_acima_10_porcentol = media + norm.ppf(0.90) * desvio_padrao
print(altura_acima_10_porcentol)



# Média dos diametros
media = 0.482
desvio_padrao = 0.004
# Valor de diametro para considerar defeituoso
limite_superior = 0.491
limite_inferior = 0.473
# Calculando os limites
z_superior = (limite_superior - media) / desvio_padrao
z_inferior = (limite_inferior - media) / desvio_padrao
# Calculando as probabilidades usando a funçao de distribuição cumulativa (CDF)
probabilidade_superior = 1 - norm.cdf(z_superior)
probabilidade_inferior = norm.cdf(z_inferior)
# Porcentagem de peças defeituosas
porcentagem_defeituosas = (probabilidade_superior + probabilidade_inferior)
print(porcentagem_defeituosas)




#experimento de Buffon

# intervalos para x e y
x_intervalo = (0, math.pi) 
y_intervalo = (0, 1/2)  

#numero de pontos 
N = 10000

# numeros aleatorios dentro dos intervalos para x (theta) e y
theta_valores = np.random.uniform(x_intervalo[0], x_intervalo[1], N)
y_valores = np.random.uniform(y_intervalo[0], y_intervalo[1], N)

#valores de x (theta) para a funçao seno
theta_seno = np.linspace(x_intervalo[0], x_intervalo[1], 100)
# valores de y correspondentes a função seno
y_seno = 0.25 * np.sin(theta_seno)

# pontos abaixo da curva da funçao seno
pontos_abaixo_curva = [(theta_valores[i], y_valores[i]) for i in range(N) if y_valores[i] < 0.25 * np.sin(theta_valores[i])]

# razao entre o numero total e o numero de dados abaixo da curva
razao = N/len(pontos_abaixo_curva)

# Plotar os pontos em um grafico de dispersao
#plt.scatter(theta_valores, y_valores, color='blue', alpha=0.7, label=f'Pontos Aleatórios (N={N})')
# Plotar a curva da funçao seno
#plt.plot(theta_seno, y_seno, color='red')
plt.title('Pontos Aleatórios abaixo da Curva de sin(\u03B8)')
plt.xlabel('\u03B8')
plt.ylabel('y')

# Plotar os pontos abaixo da curva em verde
#for ponto in pontos_abaixo_curva:
#    plt.scatter(ponto[0], ponto[1], color='green')


plt.legend(title=f'\u03C0 = {razao:.4f}', loc='upper right', bbox_to_anchor=(1.0, 1.0), bbox_transform=plt.gcf().transFigure)
plt.grid(True)
plt.show()

#calculo da integral I
resultados = []

# Loop para diferentes valores de N
for N in range(10, 5001, 10):
    # Gerando números aleatórios dentro dos intervalos para x (theta) e y
    theta_valores = np.random.uniform(x_intervalo[0], x_intervalo[1], N)
    y_valores = np.random.uniform(y_intervalo[0], y_intervalo[1], N)
    
    # Calculando pontos abaixo da curva
    pontos_abaixo_curva = sum(1 for i in range(N) if y_valores[i] < 0.25 * np.sin(theta_valores[i]))
    
    # Calculando pi/2 * pontos abaixo da curva / N e adicionando à lista de resultados
    resultado = (math.pi / 2) * pontos_abaixo_curva / N 
    resultados.append(resultado)

media = np.mean(resultados)
rms = np.sqrt(np.mean(np.square(resultados)))

# Plotando o histograma
plt.hist(resultados, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.title(''r'$\int_{0}^{2\pi} \ \frac{1}{4} \sin(x) \, dx$')
legenda = f'N: {N}\nMédia: {media:.2f}\nRMS: {rms:.2f}'
plt.legend([legenda], loc='upper right')
plt.xlabel('Valores')
plt.ylabel('N')
plt.grid(True)
plt.show()
