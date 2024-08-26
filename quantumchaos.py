import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parámetros del sistema
hbar = 1.0  # Constante de Planck reducida
omega = 1.0  # Frecuencia del oscilador armónico
k = 0.2  # Constante de acoplamiento

# Matrices de Pauli (operadores cuánticos básicos)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

# Hamiltoniano del sistema cuántico
H = omega * (np.kron(sigma_z, sigma_z) + k * np.kron(sigma_x, sigma_x))

# Función para calcular la evolución temporal del sistema cuántico
def evolucion_cuantica(H, psi0, t):
    U = expm(-1j * H * t / hbar)  # Operador de evolución temporal
    psi_t = np.dot(U, psi0)  # Estado en el tiempo t
    return psi_t

# Estado inicial de las partículas (entrelazado)
psi0 = 1/np.sqrt(2) * (np.kron([1, 0], [0, 1]) + np.kron([0, 1], [1, 0]))

# Tiempo de simulación
T = np.linspace(0, 50, 500)

# Almacenar las probabilidades a lo largo del tiempo
probabilidades = []

# Simulación
for t in T:
    psi_t = evolucion_cuantica(H, psi0, t)
    # Probabilidad de encontrar el sistema en el estado base
    prob = np.abs(np.dot(psi0.conj().T, psi_t))**2
    probabilidades.append(prob)

# Conversión a un array numpy
probabilidades = np.array(probabilidades)

# Plotear la evolución de la probabilidad en función del tiempo
plt.figure(figsize=(10, 6))
plt.plot(T, probabilidades, label='Probabilidad del estado base')
plt.xlabel('Tiempo')
plt.ylabel('Probabilidad')
plt.title('Evolución temporal del sistema cuántico entrelazado')
plt.legend()
plt.grid(True)
plt.show()

# Representación del caos: Sensibilidad a las condiciones iniciales
# Variamos ligeramente la condición inicial y observamos la diferencia en la evolución
delta_psi = 1e-3 * np.random.randn(*psi0.shape)
psi0_prime = psi0 + delta_psi

# Evolución del estado perturbado
probabilidades_prime = []

for t in T:
    psi_t_prime = evolucion_cuantica(H, psi0_prime, t)
    prob_prime = np.abs(np.dot(psi0_prime.conj().T, psi_t_prime))**2
    probabilidades_prime.append(prob_prime)

probabilidades_prime = np.array(probabilidades_prime)

# Plotear ambas evoluciones para visualizar el caos
plt.figure(figsize=(10, 6))
plt.plot(T, probabilidades, label='Estado inicial original')
plt.plot(T, probabilidades_prime, label='Estado inicial perturbado', linestyle='dashed')
plt.xlabel('Tiempo')
plt.ylabel('Probabilidad')
plt.title('Teoría del Caos en Sistemas Cuánticos Entrelazados')
plt.legend()
plt.grid(True)
plt.show()