<img width="1024" height="172" alt="retouch_2025111217390816" src="https://github.com/user-attachments/assets/62278260-84c7-4e9d-843d-1014484c471d" />

<div align="center">
⚛️ QuoreMind v1.0.0
  
----
<h1>Sistema Metripléctico Cuántico-Bayesiano</h1>

![Software](https://img.shields.io/badge/Hybrid-QuoreMind%201.0-white)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-Apache%202.0-green)
![Software](https://img.shields.io/badge/Quantum-AI%20SmokApp-black)
</div>

----

## 🧭 Visión General

**QuoreMind** es un framework analítico de vanguardia diseñado para modelar y predecir la evolución de sistemas cuánticos abiertos mediante la integración de:

- 🔷 **Dinámica Metripléctica** (reversible + disipativa)
- 🔴 **Lógica Bayesiana Cuántica**
- 🟢 **Ruido Probabilístico de Referencia (PRN)**
- ⚪ **Entropía de von Neumann**
- 🔵 **Corchetes de Poisson**

### Aplicaciones Clave

✅ **Detección forense de anomalías** en información cuántica  
✅ **Mitigación de ataques HN/DL** (Harvest Now, Decrypt Later)  
✅ **Modulación de fase cuasiperiódica** para criptografía dinámica  
✅ **Análisis de decoherencia cuántica** y entrelazamiento  
✅ **Optimización de estados cuánticos** resistentes a ruido  

---

## ✨ Características Clave

### 🔷 Estructura Nativamente Cuántica

| Característica | Descripción | Impacto en Seguridad |
|---|---|---|
| **Operador Áureo** `Ô_n` | Modula fase cuasiperiódica y paridad del sistema | Ancla estados a secuencia no trivial → Cifrado dinámico robusto |
| **Entropía von Neumann** `S(ρ)` | Métrica fundamental para medir desorden y entrelazamiento | Base para cuantificar decoherencia esperada vs. anómala |
| **Distancia Mahalanobis Cuántica** `D_M` | Desviación estructural respecto a PRN esperado | D_M alta → Indicador potencial de intrusión |

### 🔶 Arquitectura Metripléctica

Fusiona reversibilidad y disipación (análogo a Ecuación de Lindblad):

```
df/dt = {f, H} + [f, S]_M

Parte reversible: {f, H}     (Corchetes de Poisson)
Parte disipativa:  [f, S]_M  (Matriz métrica M)
```

| Componente | Función |
|---|---|
| **Corchetes de Poisson** | Dinámica reversible (Hamiltoniana) |
| **Matriz Métrica M** | Modela disipación e irreversibilidad |

### 🔴 Lógica Bayesiana y PRN

- **PRN**: Modela ruido ambiental estocástico esperado
- **Inferencia Bayesiana**: Calcula probabilidad posterior para decisiones binarias óptimas
- **Coherencia Dinámica**: Parámetro adaptativo basado en estado del sistema

---

## 🏗️ Arquitectura del Proyecto

### Estructura de Clases

```
VonNeumannEntropy
    ├─ compute_von_neumann_entropy()      [Cálculo cuántico]
    ├─ density_matrix_from_state()        [Construcción ρ]
    └─ mixed_state_entropy()              [Mezclas estadísticas]

PoissonBrackets
    ├─ poisson_bracket()                  [Estructura simpléctica]
    └─ liouville_evolution()              [Ecuación de Liouville]

MetriplecticStructure
    ├─ metriplectic_bracket()             [Corchete metriplexico]
    └─ metriplectic_evolution()           [Evolución híbrida]

BayesLogic
    ├─ calculate_posterior_probability()  [Teorema de Bayes]
    ├─ calculate_joint_probability()      [Probabilidades conjuntas]
    └─ calculate_probabilities_and_select_action()  [Decisión final]

QuantumBayesMahalanobis (extends BayesLogic)
    ├─ compute_quantum_mahalanobis()      [Distancia vectorizada]
    ├─ quantum_cosine_projection()        [Proyecciones coseno]
    └─ predict_quantum_state()            [Predicción de estado]

PRN / EnhancedPRN
    ├─ adjust_influence()                 [Modulación de ruido]
    ├─ combine_with()                     [Combinación de PRN]
    └─ record_quantum_noise()             [Registro de anomalías]

QuantumNoiseCollapse (core)
    ├─ simulate_wave_collapse_metriplectic()  [Simulación principal]
    ├─ objective_function_with_noise()   [Función objetivo]
    └─ optimize_quantum_state()          [Optimización Adam]
```

---

## 🚀 Instalación y Requerimientos

### Requisitos Previos
- **Python 3.9+**
- **pip** o **conda**

### Instalación

```bash
# Clonar repositorio
git clone https://github.com/tlacaelel666/QuoreMind-Metiplectic.git
cd quoremind

# Instalar dependencias
pip install numpy tensorflow tensorflow-probability scikit-learn scipy

# (Opcional) Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### Dependencias

```python
numpy >= 1.21.0
tensorflow >= 2.10.0
tensorflow-probability >= 0.19.0
scikit-learn >= 1.0.0
scipy >= 1.7.0
```

---

## 📖 Uso Básico

### Ejemplo 1: Cálculo de Entropía von Neumann

```python
from quoremind import VonNeumannEntropy
import numpy as np

# Crear estado puro de Bell
state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
density_matrix = VonNeumannEntropy.density_matrix_from_state(state)

# Calcular entropía (normalizada a [0, 1])
entropy = VonNeumannEntropy.compute_von_neumann_entropy(
    density_matrix,
    state # sigmoid, tanh, log_compression, min_max, clamp
)

print(f"Entropía von Neumann: {entropy:.6f}")  # Debe oscilar ∈ [0, 1] 
```

### Ejemplo 2: Análisis de Corchetes de Poisson

```python
from quoremind import PoissonBrackets
import numpy as np

# Definir Hamiltoniano
H = lambda q, p: 0.5 * p**2 + 0.5 * q**2  # Oscilador armónico

# Definir observable
x = lambda q, p: q

# Calcular corchete de Poisson
q_val = np.array([1.0])
p_val = np.array([1.0])

bracket = PoissonBrackets.poisson_bracket(x, H, q_val, p_val)
print(f"{{x, H}} = {bracket:.6f}")  # Debe ≈ p = 1.0

# Evolución de Liouville: dx/dt = {x, H}
df_dt = PoissonBrackets.liouville_evolution(H, x, q_val, p_val)
print(f"dx/dt = {df_dt:.6f}")
```

### Ejemplo 3: Simulación de Colapso Metripléctico (Uso Completo)

```python
from quoremind import (
    QuantumNoiseCollapse,
    VonNeumannEntropy
)
import numpy as np

# Inicializar sistema
collapse_system = QuantumNoiseCollapse(prn_influence=0.6)

# Crear estado cuántico de prueba
state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
density_matrix = VonNeumannEntropy.density_matrix_from_state(state)
quantum_states = np.random.randn(10, 2)

# Matriz métrica (disipación)
M = np.array([[0.1, 0.0], [0.0, 0.1]])

# Simular colapso con estructura metripléctica
result = collapse_system.simulate_wave_collapse_metriplectic(
    quantum_states=quantum_states,
    density_matrix=density_matrix,
    prn_influence=0.6,
    previous_action=1,
    M=M
)

# Resultados
print(f"✓ Estado colapsado: {result['collapsed_state']:.6f}")
print(f"✓ Acción bayesiana: {result['action']}")
print(f"✓ Entropía Shannon (norm): {result['shannon_entropy_normalized']:.6f}")
print(f"✓ Entropía von Neumann: {result['von_neumann_entropy']:.6f}")
print(f"✓ Coherencia: {result['coherence']:.6f}")
print(f"✓ Distancia Mahalanobis: {result['mahalanobis_normalized']:.6f}")
print(f"✓ Evolución metripléctica: {result['metriplectic_evolution']:.6f}")
print(f"✓ Posterior bayesiana: {result['bayesian_posterior']:.6f}")
```

### Ejemplo 4: Optimización de Estados Cuánticos

```python
from quoremind import QuantumNoiseCollapse
import numpy as np

# Inicializar
collapse_system = QuantumNoiseCollapse(prn_influence=0.6)

# Estados iniciales aleatorios
initial_states = np.random.randn(5, 2)
target_state = np.array([0.8, 0.2])

# Optimizar hacia estado objetivo
optimized_states, final_loss = collapse_system.optimize_quantum_state(
    initial_states=initial_states,
    target_state=target_state,
    max_iterations=100,
    learning_rate=0.01
)

print(f"✓ Pérdida final: {final_loss:.6f}")
print(f"✓ Estados optimizados:\n{optimized_states}")
```

### Ejemplo 5: Análisis de Anomalías con Mahalanobis

```python
from quoremind import QuantumBayesMahalanobis
import numpy as np

# Inicializar
analyzer = QuantumBayesMahalanobis()

# Estados de referencia (distribución normal)
reference_states = np.random.randn(100, 2)

# Estados anómalos
anomalous_states = np.random.randn(10, 2) + np.array([3.0, 3.0])

# Calcular distancias de Mahalanobis
distances = analyzer.compute_quantum_mahalanobis(
    reference_states,
    anomalous_states
)

print(f"Distancias Mahalanobis (anomalías):")
for i, d in enumerate(distances):
    print(f"  Estado {i}: {d:.4f}")

# Umbral de detección (ejemplo: 3σ)
threshold = np.mean(distances) + 3 * np.std(distances)
anomalies = distances > threshold

print(f"\n✓ Anomalías detectadas: {np.sum(anomalies)}/{len(anomalies)}")
```

---

## 📊 Métricas y Normalizaciones

### Normalización de Entropía

El framework ofrece **5 métodos** para normalizar entropía a `[0, 1]`:

| Método | Fórmula | Caso de Uso |
|--------|---------|-----------|
| **sigmoid** | `1/(1+e^-S)` | ✅ Recomendado (suave, diferenciable) |
| **tanh** | `(tanh(S/2)+1)/2` | Simétrico alrededor de 0.5 |
| **log_compression** | `log(1+S)/log(1+max_S)` | Física estadística |
| **min_max** | `S/log(dim)` | Teórico puro |
| **clamp** | `min(S/max, 1.0)` | Rápido/simple |

### Parámetros de Configuración

```python
from quoremind_v1_0_0 import BayesLogicConfig

config = BayesLogicConfig(
    epsilon=1e-6,
    high_entropy_threshold=0.8,      # Umbral de entropía alta
    high_coherence_threshold=0.6,    # Umbral de coherencia alta
    action_threshold=0.5              # Umbral para acción bayesiana
)
```

---

## 🔍 Validación y Testing

El framework incluye validación automática:

```python
# Validación de estructura metripléctica
# ✓ Ecuaciones de Hamilton se satisfacen
# ✓ dS/dt > 0 (producción de entropía positiva)
# ✓ Conservación de energía (parte reversible)

# Validación de convergencia (Adam)
# ✓ Loss disminuye monotónicamente
# ✓ Gradientes no explotan
# ✓ Estados convergen a objetivo
```

---

## 🧬 Ecuaciones Fundamentales

### Entropía de von Neumann
```
S(ρ) = -Tr(ρ log ρ) = -Σ λᵢ log λᵢ
```

### Corchetes de Poisson
```
{f, g} = (∂f/∂q)(∂g/∂p) - (∂f/∂p)(∂g/∂q)
```

### Ecuación de Liouville
```
df/dt = {f, H}
```

### Estructura Metripléctica
```
df/dt = {f, H} + [f, S]_M

donde:
- {f, H}: parte reversible (Hamiltoniana)
- [f, S]_M: parte disipativa (matriz métrica M)
```

### Distancia de Mahalanobis
```
D_M = √[(x - μ)ᵀ Σ⁻¹ (x - μ)]
```

### Teorema de Bayes
```
P(A|B) = P(B|A) × P(A) / P(B)
```

---

## 🎯 Casos de Uso Principales

### 1. **Detección de Anomalías Cuánticas**
```python
# Detectar intrusión mediante Mahalanobis anómala
distances = analyzer.compute_quantum_mahalanobis(
    reference_states,
    observed_states
)
anomaly_detected = (distances > threshold).any()
```

### 2. **Monitoreo de Decoherencia**
```python
# Rastrear decoherencia esperada vs. anómala
for cycle in range(n_cycles):
    result = collapse_system.simulate_wave_collapse_metriplectic(...)
    entropy = result['shannon_entropy_normalized']
    mahal = result['mahalanobis_normalized']
    
    # Si ambos son anormalmente altos → posible ataque
    if entropy > 0.9 and mahal > 0.8:
        log_alert("INTRUSION DETECTED")
```

### 3. **Optimización de Cifrado Dinámico**
```python
# Generar estados objetivo resistentes a ruido
target = generate_secure_state()
optimized, loss = collapse_system.optimize_quantum_state(
    initial_states=random_states,
    target_state=target,
    max_iterations=200
)
# Los estados optimizados resisten interferencia
```

### 4. **Análisis Forense**
```python
# Estimar parámetro de no-localidad λ desde D_M anómala
lambda_estimate = estimate_nonlocality(anomalous_distances)
# Documentar en log forense
```

---

## 📈 Rendimiento y Complejidad

| Operación | Complejidad | Tiempo (aprox.) |
|-----------|------------|-----------------|
| Entropía von Neumann | O(n³) | ~0.1ms (n=2) |
| Corchete de Poisson | O(1) | ~0.05ms |
| Mahalanobis (vectorizado) | O(nm²) | ~1ms (n=100, m=2) |
| Optimización (100 iter) | O(nm²·iter) | ~500ms |

---

## 🐛 Troubleshooting

### Error: `ValueError: Argumento entropy debe estar entre 0.0 y 1.0`
**Solución**: Usar normalización automática (ya implementada)
```python
entropy_norm = 1.0 / (1.0 + np.exp(-entropy))
```

### Error: Matriz de covarianza singular
**Solución**: El código usa pseudo-inversa automáticamente
```python
inv_cov = np.linalg.pinv(cov_matrix)  # Pseudo-inversa
```

### Convergencia lenta en optimización
**Solución**: Aumentar learning_rate o max_iterations
```python
optimized, loss = collapse_system.optimize_quantum_state(
    initial_states=states,
    target_state=target,
    max_iterations=500,    # ← Aumentar
    learning_rate=0.05     # ← Aumentar
)
```

---

## 📚 Referencias y Documentación

- **Dinámica Metripléctica**: Morrison, P. J. (1986). "Structural, Hamiltonian, and Lagrangian Formulation"
- **Entropía von Neumann**: von Neumann, J. (1932). "Mathematical Foundations of QM"
- **Ecuación de Lindblad**: Lindblad, G. (1976). "On the Generators of QDynamical Semigroups"
- **Distancia Mahalanobis**: Mahalanobis, P. C. (1936). "On the Generalized Distance"
- **Lógica Bayesiana**: Bayes, T. (1763). "Essay Towards Solving a Problem"

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas, especialmente en:

- 🔹 **Integración de Polaridad del Vacío** `η(r)` como modulador de M
- 🔹 **Rastreo Forense Avanzado**: Estimación de `λ` desde anomalías
- 🔹 **Quantum Machine Learning**: Optimización de función objetivo con QML
- 🔹 **GPU Acceleration**: Vectorización CUDA/ROCm
- 🔹 **Interfaz Gráfica**: Dashboard en tiempo real de métricas

---

## 📄 Licencia

Este proyecto está distribuido bajo la licencia **Apache 2.0**.

```
Copyright 2025 Jacobo Tlacaelel Mina Rodríguez
Licensed under the Apache License, Version 2.0
```

Ver [LICENSE](LICENSE) para detalles completos.

---

## 📞 Contacto y Soporte

- **Autor**: Jacobo Tlacaelel Mina Rodríguez
- **Email**: jakocrazykings@gmail.com
- **Issues**: [GitHub Issues](https://github.com/smokeappstore/QuoreMind-Metriplectic/issues)
- **Documentación**: [Wiki](https://github.com/smokeappstore/QuoreMind-Metriplectic/wiki)

---

## 🎓 Cita Académica

Si usas QuoreMind en investigación, por favor cita:

```bibtex
@software{quoremind2025,
  title={QuoreMind v1.0.0: Sistema Metripléctico Cuántico-Bayesiano},
  author={Mina Rodríguez, Jacobo Tlacaelel},
  year={2025},
  url={https://github.com/smokeappstore/QuoreMind-Metriplectic},
  license={Apache-2.0}
}
```

---
<div align="center">
<h1> Last Updated: Noviembre 2025  </h1>
Version: 1.0.0  
Status: ✅ Production Ready 
</div>
