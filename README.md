# Trabajo práctico: agentes para el 2048 simplificado

En este trabajo van a programar dos agentes que juegan una versión reducida del 2048:

- uno que **planifica** buscando en el árbol de juego, con **minimax**;
- otro que **aprende jugando**, con **Q-learning**, sin conocer las probabilidades del juego.

El código de base ya implementa el juego, la simulación de partidas, la estructura de los agentes y los
scripts para evaluarlos. Cada equipo/grupo debe completar sólo las partes marcadas con `TODO` en dos archivos.

## El juego

Usamos las reglas del 2048 original sobre un tablero chico, de **2 × 2** por defecto:

- Hay cuatro movimientos: `Left`, `Right`, `Up` y `Down`. Todas las fichas se deslizan hasta el borde.
- Dos fichas iguales que chocan se fusionan en una: 2 + 2 = 4. Cada ficha se fusiona a lo sumo una vez
  por jugada.
- La **recompensa** de una jugada es la suma de las fichas creadas al fusionar. La puntuación de una
  partida es la suma de sus recompensas.
- Sólo son legales las jugadas que cambian el tablero. Si no queda ninguna, la partida termina.
- Después de cada jugada aparece una ficha nueva en una casilla vacía elegida al azar: un 2 con
  probabilidad 0,9 o un 4 con probabilidad 0,1.
- La partida empieza con dos fichas al azar.

Cada jugada tiene una **parte determinista**, deslizar y fusionar, que produce un tablero intermedio
(el *afterstate*), y una **parte aleatoria**, la ficha nueva:

```
 tablero     Left, +4 puntos     afterstate     ficha nueva     siguiente estado posible
 | 2 2 |                          | 4 . |                         | 4 2 |  | 4 4 |  | 4 . |  | 4 . |
 | . 4 |     -------------->      | 4 . |     ------------>       | 4 . |  | 4 . |  | 4 2 |  | 4 4 |
                                                                   0,45     0,05     0,45     0,05
```

En el tablero de 2 × 2 hay 662 estados alcanzables, así que una tabla Q cabe entera en memoria. El
tamaño del tablero es configurable (`Game2048Problem(rows, cols)`) para los experimentos de la parte 3.

## Organización del código

La estructura sigue la del framework de búsqueda que usamos en los trabajos anteriores:

```
core/
    game_state.py          GameState: estado hashable (como SearchState)
    stochastic_problem.py  StochasticProblem: el juego como proceso de decisión de Markov (como SearchProblem)
    agent.py               Agent: un agente elige acciones (como SearchAlgorithm)
    simulation.py          play_game y evaluate_agent: jugar y evaluar partidas
problems/
    game_2048_state.py     Game2048State: el tablero
    game_2048_problem.py   Game2048Problem: las reglas del 2048
algorithms/
    random_agent.py        RandomAgent: juega al azar (baseline)
    minimax_agent.py       MinimaxAgent                        <- Parte 1
    q_learning_agent.py    QLearningAgent                      <- Parte 2
tests/                     tests (pruebas) automáticas
play_2048.py               juega una partida mostrando cada jugada
evaluate_agents.py         compara los agentes en muchas partidas
train_q_learning.py        entrena Q-learning y muestra su curva de aprendizaje
INFORME.md                 plantilla del informe (Parte 3)
```

`StochasticProblem` ofrece dos formas de usar el juego:

| Método | Qué hace | Quién puede usarlo |
|---|---|---|
| `get_actions(s)` | acciones legales de `s` (vacía si `s` es final) | todos |
| `is_terminal(s)` | si la partida terminó | todos |
| `step(s, a, rng)` | juega `a` y devuelve `(recompensa, siguiente_estado, fin)` | todos |
| `apply_action(s, a)` | parte determinista: `(afterstate, recompensa)` | sólo los agentes de búsqueda |
| `get_chance_outcomes(t)` | todos los resultados posibles de la ficha nueva, `[(estado, probabilidad), ...]` | sólo los agentes de búsqueda |
| `heuristic(s)` | estimación de lo que queda por ganar desde `s` (la suma de las fichas) | sólo los agentes de búsqueda |

Los agentes de búsqueda conocen el modelo completo del juego. El agente de Q-learning **no**: sólo
puede actuar con `step` y consultar las acciones legales.

## Cómo ejecutarlo

Sólo se necesita una versión de Python 3 (fue testeado con Python 3.8, 3.8 o posterior es aceptable); no hay dependencias externas. Desde la carpeta del proyecto:

```
python -m unittest discover -s tests -v     # todas las pruebas
python evaluate_agents.py                   # compara los agentes implementados
python train_q_learning.py                  # curva de aprendizaje de Q-learning
python play_2048.py                         # una partida, jugada por jugada
```

Al principio sólo pasan las pruebas de `tests/test_game_2048.py`, que verifican el juego. Las demás
fallan con `NotImplementedError` hasta que completen el código.

## Consigna

### Parte 1: minimax (obligatoria)

Completen `max_value` y `chance_value` en `algorithms/minimax_agent.py`.

El árbol de juego alterna dos tipos de nodos. En los nodos **MAX** el jugador elige la jugada; en los
nodos **de azar** el ambiente coloca la ficha nueva. Minimax trata al ambiente como un adversario: supone
que la ficha aparece siempre en el peor lugar posible para el jugador.

El valor de un estado cuenta las recompensas que se obtendrán a partir de él, mirando `d` jugadas hacia
adelante:

$$
V(s, d) =
\begin{cases}
0 & \text{si } s \text{ es final} \\
h(s) & \text{si } d = 0 \\
\max_{a} \big[\, r(s, a) + C(t_{s,a},\, d) \,\big] & \text{en otro caso}
\end{cases}
\qquad\qquad
C(t, d) = \min_{s' \in \text{resultados}(t)} V(s', d - 1)
$$

donde $t_{s,a}$ es el afterstate que produce la jugada $a$ en $s$, $r(s, a)$ su recompensa y $h$ la
heurística del problema. `max_value` calcula $V$ y `chance_value` calcula $C$. El método
`choose_action`, ya implementado, elige la jugada que maximiza $r(s, a) + C(t_{s,a}, d)$.

No borren la línea `self.expanded_nodes += 1` de `max_value`: cuenta los nodos expandidos y la van a
usar en la parte 3.

Las pruebas de esta parte están en `tests/test_minimax.py` (clase `TestMinimax`).

### Parte 2: Q-learning (obligatoria)

Completen `epsilon_greedy_action` y `update` en `algorithms/q_learning_agent.py`. El ciclo de
entrenamiento, `train`, ya está implementado y llama a esos dos métodos.

- `epsilon_greedy_action(s)`: con probabilidad ε, una acción legal elegida al azar de manera uniforme;
  si no, la mejor según Q (pueden usar `best_action`). Usen `self.rng` para todo lo aleatorio, así los
  resultados son reproducibles.
- `update(s, a, r, s', fin)`: la regla de Q-learning

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \big[\, y - Q(s, a) \,\big]
\qquad
y =
\begin{cases}
r & \text{si } s' \text{ es final} \\
r + \gamma \max_{a' \in \text{legales}(s')} Q(s', a') & \text{en otro caso}
\end{cases}
$$

La tabla Q es el diccionario `self.q`, con claves `(estado, acción)`; los pares que no están valen 0
(`get_q` ya lo resuelve). El agente no debe llamar a `get_chance_outcomes`, `apply_action` ni
`heuristic`: tiene que aprender sólo de lo que observa.

Las pruebas de esta parte están en `tests/test_q_learning.py`. Una de ellas entrena 5000 episodios y
exige que la política aprendida supere los 50 puntos de media.

### Parte 3: experimentos e informe (obligatoria)

Respondan en `INFORME.md`. Incluyan las tablas o salidas que usen para justificar cada respuesta.

1. **Comparación.** Con `evaluate_agents.py`, que evalúa minimax con cada profundidad de la lista
   `DEPTHS`, completen una tabla con la puntuación media y la distribución de la ficha máxima de: el
   agente al azar, minimax con profundidad 1, 2 y 3, y Q-learning tras 30 000 episodios. Agreguen el
   tiempo que tarda cada uno.
2. **Minimax frente al azar real.** ¿Por qué minimax obtiene menos puntos que Q-learning, aunque conoce
   el juego completo y Q-learning no? Piensen qué supone minimax sobre la ficha nueva y si eso es
   realista. Busquen, o construyan a mano, un tablero en el que esa suposición lo lleve a elegir peor.
3. **Profundidad.** Midan los nodos expandidos por jugada para profundidades 1, 2 y 3. `evaluate_agents.py`
   los muestra para minimax: es `expanded_nodes` dividido por la cantidad total de jugadas. ¿Cómo crece?
   ¿Mejora la puntuación al aumentar la profundidad? Relaciónenlo con lo visto en clase sobre la recursión
   sin memoria y los subproblemas que se repiten.
4. **Parámetros de Q-learning.** Con `train_q_learning.py`, comparen las curvas de aprendizaje para al
   menos dos valores de α y dos de ε. ¿Qué efecto tiene cada uno?
5. **Un tablero más grande.** Repitan minimax con profundidad 1 y Q-learning con 30 000 episodios en un
   tablero de 3 × 3. En `evaluate_agents.py` pongan `ROWS, COLS = 3, 3`, `DEPTHS = [1]` y `GAMES = 200`:
   con profundidad 2 o más, minimax tarda demasiado en 3 × 3. Para ver cuántas entradas tiene la tabla Q,
   pongan también `ROWS, COLS = 3, 3` en `train_q_learning.py`. Comparen con el agente al azar. ¿Por qué
   el Q-learning con tabla rinde mucho peor que en 2 × 2?

## Valores de referencia

Con una implementación correcta, en el tablero de 2 × 2, `evaluate_agents.py` (1000 partidas, semilla 7)
da valores cercanos a estos. Pueden variar un poco según el orden en que se use el generador aleatorio.

| Agente | Puntuación media |
|---|---|
| Al azar | ≈ 19 |
| Minimax, profundidad 2 | ≈ 58 |
| Q-learning, 30 000 episodios | ≈ 66 |
| Política óptima (valor exacto) | 66,96 |

El valor óptimo se calculó resolviendo el juego completo con el modelo, por programación dinámica.

## Entrega

- **Qué entregar:** el repositorio con `algorithms/minimax_agent.py` y `algorithms/q_learning_agent.py`
  completos e `INFORME.md` respondido.
- **Reglas:**
  - no modifiquen `core/`, `problems/` ni `tests/`; si necesitan otro comportamiento, escriban subclases
    en archivos nuevos;
  - no agreguen dependencias externas;
  - el agente de Q-learning no puede usar el modelo del juego.
- **Cómo se evalúa:** que pasen todas las pruebas (`python -m unittest discover -s tests`), que los
  resultados sean razonables comparados con los de referencia y la calidad de las respuestas del informe.
- **Fecha y forma de entrega:** Se entrega clonando el repositorio y enviando por email el enlace al repositorio del grupo. Se puede resolver en grupos de 3-5 integrantes. La fecha de entrega es flexible, pero se espera que lo completen en aproximadamente un mes. 
