# Aprendizaje por refuerzo tabular, y qué significa realmente "óptimo"

[![CI](https://github.com/JosElias23/rl-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/JosElias23/rl-from-scratch/actions/workflows/ci.yml)
[![tests](https://img.shields.io/badge/tests-37%20passing-brightgreen)](https://github.com/JosElias23/rl-from-scratch/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

[English](README.md) · **Español**

Q-learning y SARSA escritos desde primeros principios, sin ninguna librería de RL
en ninguna parte, y medidos contra óptimos calculados de forma exacta por
programación dinámica en lugar de contra cifras citadas de internet.

Los algoritmos son la parte fácil. El trabajo interesante está en decidir contra
qué compararlos, y el resultado principal de este proyecto es que la cifra que
todos citan como "el óptimo" de FrozenLake responde a otra pregunta.

---

## Resultados

### FrozenLake 4x4, con deslizamiento, 5.000 episodios de evaluación greedy

| Agente | Tasa de éxito | IC 95 % | % del óptimo exacto | Coincidencia con la política óptima |
|---|---:|:--|---:|---:|
| Aleatorio | 0,0120 | — | 1,6 % | — |
| SARSA | 0,7378 | [0,7254, 0,7498] | 99,1 % | **100 %** |
| Q-learning | 0,7378 | [0,7254, 0,7498] | 99,1 % | **100 %** |
| _Óptimo exacto (inducción hacia atrás)_ | _0,7442_ | _exacto_ | _100 %_ | — |

![Resultados de FrozenLake](reports/figures/frozenlake_results.png)

Ambos agentes recuperan exactamente la política **estacionaria** óptima. Cada uno
de los 16 estados recibe la acción que eligió la iteración de valor. Su tasa de
éxito medida coincide con lo que esa política obtiene al simularla, hasta el
cuarto decimal.

Los 0,64 puntos porcentuales restantes no son una falla de entrenamiento. Son
exactamente el valor de saber cuánto tiempo queda, algo que ninguna política
estacionaria puede representar. Más sobre eso abajo.

---

## El hallazgo principal: "el óptimo" son tres números distintos

![Los tres techos](reports/figures/frozenlake_ceiling.png)

La cifra que se cita habitualmente para el juego óptimo en FrozenLake 4x4 con
deslizamiento es cerca de **74 %**. Ese número no es una propiedad del entorno.
La iteración de valor sobre el modelo de transición publicado da:

| Pregunta que se está haciendo | Respuesta exacta |
|---|---:|
| Lo mejor alcanzable, sin límite de tiempo | **0,8235** (= 14/17) |
| Lo mejor alcanzable dentro de 100 pasos, con una política que puede mirar el reloj | **0,7442** |
| La política óptima sin límite, interrumpida a los 100 pasos | 0,7402 |

Las tres son correctas. Responden preguntas distintas, y citar una sin nombrar el
horizonte no significa nada.

Por qué el límite de tiempo cuesta tanto. La política óptima es deliberadamente
lenta. Se pega a los muros y acepta deslizamientos laterales para que ningún
resbalón desafortunado la empuje a un hoyo, lo que significa que cruzar una
grilla de 4×4 muchas veces toma bastante más de 100 pasos. El `TimeLimit` por
defecto de Gymnasium trunca esos éxitos que sí habrían llegado:

```
horizonte    50   ->  0,5356
horizonte   100   ->  0,7402      <- el "~74 %" que se cita habitualmente
horizonte   200   ->  0,8164
horizonte   500   ->  0,8235
sin límite        ->  0,8235
```

**Por qué una política estacionaria no puede alcanzar ni siquiera el óptimo de
100 pasos.** Una política que sabe que le quedan diez pasos debería dejar de
jugar a la segura y apostar por una carrera directa. El óptimo de un objetivo
con límite de tiempo es, por lo tanto, *no estacionario*, y la inducción hacia
atrás lo encuentra:

```
V_0(s) = 0
V_k(s) = max_a  sum_s'  P(s'|s,a) [ r(s,a,s') + V_{k-1}(s') ]
```

Esa política obtiene 0,7442 frente al 0,7402 de la política estacionaria. El
comportamiento se ve directamente: desde el estado inicial, con 100, 50 o 20
pasos restantes juega la acción `Left`; con 10 restantes cambia a `Right` y se
lanza a ganar.

Ambas cifras están verificadas contra 20.000 episodios simulados, y CI recalcula
las tres en cada push. `tests/test_planning.py` verifica que la inducción hacia
atrás coincide con simular la política resultante. Si el techo estuviera mal,
cada porcentaje de este README estaría mal junto con él.

> **Esto corrige una afirmación que yo mismo había hecho sobre mi propio
> trabajo.** Una viñeta de mi CV describía el 73 % en esta tarea como "casi el
> óptimo teórico (~74 %)". La cifra de 74 % es el techo a 100 pasos, y el óptimo
> teórico es 82,35 %. La afirmación correcta es la más fuerte: el agente
> recupera exactamente la política estacionaria óptima y obtiene el 99,1 % de lo
> que cualquier política podría lograr bajo el mismo límite de tiempo.

El registro de decisiones, incluido el resultado de una sola seed que hubo que
corregir, está en [`docs/DECISIONS.md`](docs/DECISIONS.md).

---

## CartPole: lo que cuesta un espacio de estados continuo

CartPole no tiene un espacio de estados finito, así que las cuatro observaciones
de valor real se cuantizan en una tabla. Esa cuantización es todo el problema de
ingeniería.

Cada agente entrenado acá supera cómodamente la barra clásica de "resuelto" de
195 de retorno medio, y las mejores configuraciones quedan en el techo de
truncamiento de 500 pasos. La pregunta interesante, entonces, no es si los
métodos tabulares pueden resolver CartPole -- sí pueden -- sino con qué
confiabilidad, y cuánto cuesta la discretización.

### La resolución compra confiabilidad, no rendimiento medio

![Barrido de resolución](reports/figures/cartpole_resolution_sweep.png)

El argumento de manual predice una U invertida: con muy pocos bins, situaciones
distintas colapsan en una misma celda; con demasiados, ninguna celda se visita
lo suficiente como para converger. Siete resoluciones × cinco seeds cada una,
60.000 episodios por corrida:

| Bins | Celdas de la tabla | Celdas visitadas | Retorno medio | sd entre seeds | Rango | Seeds que resuelven |
|---:|---:|---:|---:|---:|:--|---:|
| 3 | 81 | 94 % | 246,4 | **177,7** | 100–500 | 2/5 |
| 4 | 256 | 82 % | 311,7 | 193,7 | 49–492 | 3/5 |
| 5 | 625 | 66 % | 428,3 | 146,7 | 166–500 | 4/5 |
| 6 | 1.296 | 64 % | 355,6 | 126,8 | 167–500 | 4/5 |
| 8 | 4.096 | 46 % | 433,4 | 91,8 | 319–500 | 5/5 |
| 10 | 10.000 | 33 % | 456,6 | 59,1 | 356–500 | 5/5 |
| 12 | 20.736 | 27 % | **473,4** | **33,8** | 415–500 | 5/5 |

La segunda mitad de la historia del manual nunca llega. Con 12 bins el agente
visita el 27 % de 20.736 celdas y sigue siendo la mejor configuración probada, y
la única que nunca falla. Las celdas no visitadas resultan ser estados
*inalcanzables*, no estados descuidados: un péndulo a 20° con el carro
acelerando en sentido contrario es una configuración que la dinámica nunca
produce. Los bins más finos, en su mayoría, subdividen espacio vacío.

Lo que la resolución realmente compra es **confiabilidad**. Las medias de 4 a 12
bins son estadísticamente indistinguibles; la desviación estándar entre seeds se
desploma de 177,7 a 33,8. Una discretización gruesa no da un agente peor en
promedio; da una lotería. Con tres bins la tarea se resolvió dos veces en cinco
corridas, y en otra el puntaje fue 100.

Una versión anterior de este barrido, con una sola seed, produjo 499,8 con 3
bins, 38,7 con 4 y 500,0 con 5. Eso no es una curva; es ruido, y es la razón por
la que este experimento reporta seeds en vez de corridas.

### Q-learning contra SARSA: un no-resultado honesto

Ocho seeds cada uno con seis bins por dimensión, 60.000 episodios por corrida,
evaluación greedy sobre 300 episodios:

| Agente | Media | sd | Mediana | Rango | Seeds que resuelven |
|---|---:|---:|---:|:--|---:|
| **Q-learning** | **432,1** | 87,5 | 477,7 | 276–500 | **8/8** |
| SARSA | 302,3 | 125,6 | 325,2 | 121–500 | 6/8 |

Emparejado sobre las seeds que comparten, Q-learning lidera por **+129,8 de
retorno, IC 95 % [−15,0, +274,6]**. El intervalo cubre el cero.

Así que la respuesta honesta es que este experimento no logra separarlos. Tanto
la estimación puntual como el conteo de seeds resueltas favorecen a Q-learning, y
la dirección es consistente con la teoría. Q-learning aprende el valor de la
política greedy sin importar cómo explore, mientras que el objetivo on-policy de
SARSA lo sigue castigando por los movimientos ε-greedy que terminan un episodio.
Pero ocho seeds contra una desviación estándar cercana a 100 no dan poder
suficiente para declararlo, y un README que aquí proclamara un ganador estaría
afirmando algo que no replica.

La evidencia de eso está en la propia historia de este repositorio. Una corrida
con una sola seed dio 408,41 a Q-learning y 500,00 a SARSA, el orden inverso al
del ranking con ocho seeds. Una corrida anterior, antes de arreglar el bug del
flujo aleatorio, dio 500,00 a Q-learning y 294,07 a SARSA. Tres experimentos,
tres historias distintas, un diseño sin poder estadístico.

---

## Método

### Los algoritmos

Ambas son reglas de actualización de cuatro líneas, escritas a mano en vez de
importadas.

```
Q-learning   Q(s,a) <- Q(s,a) + a [ r + g max_a' Q(s',a') - Q(s,a) ]
SARSA        Q(s,a) <- Q(s,a) + a [ r + g       Q(s',a')  - Q(s,a) ]
```

El único término que difiere es toda la distinción: Q-learning aprende el valor
de la política greedy sin importar lo mal que explore, SARSA aprende el valor de
la política que realmente está siguiendo, errores de exploración incluidos. Un
test verifica que divergen cuando la siguiente acción es subóptima y coinciden
cuando es greedy.

### Tres bugs que este proyecto documenta

Los estados terminales no deben hacer bootstrap. `r + γ·V(s')` más allá de un
estado terminal inventa valor que no existe. En una tarea de recompensa dispersa
deja al agente seguro de sí mismo y equivocado, y nada lanza un error.

Los empates deben romperse al azar. Con una tabla inicializada en cero, todas las
acciones empatan. `argmax` devuelve siempre el índice 0, así que el agente
explora mucho más lento de lo que ε por sí solo sugiere. Un bug silencioso que
simplemente parece aprendizaje lento.

La evaluación no debe consumir el flujo aleatorio de entrenamiento. La selección
greedy de acciones igual necesita aleatoriedad para romper empates. Estaba
tomando del generador de entrenamiento, así que insertar una evaluación
periódica desplazaba cada decisión de entrenamiento posterior. Se detectó al
notar que dos corridas con las mismas seeds e hiperparámetros no coincidían:
500,00 con evaluación periódica, 288,47 sin ella. Medir al agente estaba
cambiando al agente. `tests/test_agents.py` ahora lo fija.

### Protocolo de evaluación

- **Las curvas de entrenamiento nunca se reportan como resultado.** Un agente
  ε-greedy está haciendo movimientos aleatorios a propósito; su retorno de
  entrenamiento subestima la política que ha aprendido. Cada número de acá viene
  de evaluación greedy con el aprendizaje desactivado.
- **Los episodios de evaluación se siembran desde un offset separado**, así los
  agentes se puntúan sobre realizaciones del entorno contra las que no
  entrenaron, y sobre las mismas entre ellos.
- **Las tasas de éxito llevan intervalos de Wilson**, no la aproximación normal,
  que se comporta mal con tasas lejanas de 0,5 sobre unos pocos cientos de
  episodios.
- **Los óptimos exactos vienen de programación dinámica**, y se contrastan contra
  simulación en la suite de tests y en CI.

---

## Reproducir estos números

```bash
git clone https://github.com/JosElias23/rl-from-scratch.git
cd rl-from-scratch
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

```bash
python -m pytest                     # 37 tests
python scripts/train_frozenlake.py   # ~4 minutos
python scripts/train_cartpole.py     # ~25 minutos
python scripts/sweep_resolution.py --workers 12   # 35 corridas, en paralelo
python scripts/compare_agents.py --workers 12     # 16 corridas, en paralelo
python scripts/make_figures.py
```

Los barridos reparten las corridas entre procesos; con doce workers cada uno toma
cerca de media hora. Todo lo demás toma minutos.

---

## Estructura del repositorio

```
rl-from-scratch/
├── configs/default.yaml       cada hiperparámetro que mueve un número
├── src/rlfs/
│   ├── agents.py              Q-learning, SARSA, baseline aleatorio
│   ├── planning.py            iteración de valor e inducción hacia atrás
│   ├── discretize.py          observaciones continuas a índices de tabla
│   ├── training.py            loop de entrenamiento y evaluación greedy
│   └── utils.py               seeding, configuración, reportes JSON
├── scripts/
│   ├── train_frozenlake.py    agentes contra el techo exacto
│   ├── train_cartpole.py      agentes en la tarea discretizada
│   ├── sweep_resolution.py    resolución vs rendimiento, multi-seed
│   ├── compare_agents.py      Q-learning vs SARSA, emparejado por seeds
│   └── make_figures.py        cada figura de este README
├── tests/                     37 tests
└── reports/                   métricas en JSON, figuras en PNG
```

---

## Limitaciones y próximos pasos

**Solo tabular.** Sin aproximación de funciones, así que nada de esto escala más
allá de espacios de estados de juguete. Ese es el punto del proyecto, no un
descuido, pero es la primera pregunta que debería hacer un entrevistador.

**Los hiperparámetros de CartPole no están ajustados.** La tasa de aprendizaje,
el descuento y el schedule de ε son valores estándar aplicados de forma idéntica
a ambos agentes. Justos como comparación, ciertamente no óptimos para ninguno.

**Los límites de la discretización están puestos a mano.** Gymnasium reporta las
dos dimensiones de velocidad como no acotadas, así que los rangos de acá son la
región de operación empírica de un péndulo controlado. Otra elección movería cada
número de CartPole.

**FrozenLake es de 4×4.** El mapa de 8×8 es un problema de exploración bastante
más difícil y no se aborda.

**Sin eligibility traces, sin double Q-learning, sin prioritised replay.** Cada
uno es una adición pequeña a este código y cada uno necesitaría el mismo
tratamiento multi-seed para poder decir algo honesto al respecto.

### Planificado

- Q(λ) con eligibility traces, y si sobrevive a la varianza entre seeds
- El mapa de 8×8 de FrozenLake, donde la exploración ε-greedy ingenua debería
  empezar a fallar
- Un agente tabular no estacionario en CartPole con el tiempo restante hasta el
  truncamiento dentro del estado, para probar si el resultado de FrozenLake se
  transfiere

---

## Licencia

MIT, ver [LICENSE](LICENSE).