# Component Guide

Questo documento spiega **ogni singola funzione** del progetto "Virtual World Explorer",
come si incastrano tra loro, e perché sono state scritte in quel modo.

---

## Che cosa fa questo progetto?

Immagina un **robottino** (l'agente) che si muove su una **scacchiera 7×7**.
Sulla scacchiera ci sono tre oggetti: una **sedia** (il bersaglio da raggiungere),
un **tavolo** e una **lampada** (distrattori / ostacoli).

Il robottino all'inizio non sa dove sia la sedia. Deve **imparare da solo**,
provando e sbagliando (come un bambino che impara a camminare).
Ogni volta che si avvicina alla sedia, riceve una **caramella** (ricompensa positiva).
Ogni volta che sbatte contro un ostacolo, riceve una **punizione** e resta fermo.
Alla fine, dopo tanti tentativi, impara a raggiungere la sedia **evitando** tavolo e lampada.

Il tutto è mostrato in una **finestrella grafica** (OpenGL) dove si vede il robottino
muoversi sulla griglia.

---

## Come sono organizzati i file

```
virtual-world-explorer/
├── requirements.txt          <-- cosa installare per far funzionare il progetto
├── src/
│   └── virtual_world_explorer/
│       ├── __init__.py       <-- trasforma la cartella in un "pacchetto" Python
│       ├── env.py            <-- il mondo: griglia, oggetti, regole del gioco
│       ├── agent.py          <-- il robottino: come impara e decide cosa fare
│       ├── detector.py       <-- gli "occhi" del robottino: dove sono gli oggetti?
│       ├── render.py         <-- il pittore: disegna la griglia nella finestra
│       └── main.py           <-- il regista: coordina tutto quanto
└── docs/
    └── component-guide.md    <-- questo file
```

---

## `requirements.txt` — Le dipendenze

Solo 2 librerie esterne:

| Libreria | A cosa serve |
|----------|-------------|
| `PyOpenGL` | Disegna la finestra con i quadratini colorati |
| `glfw` | Crea la finestra, gestisce mouse/tastiera |

Per installarle: `pip install -r requirements.txt`

---

## `env.py` — L'ambiente (il "mondo" del gioco)

Questo file descrive **la scacchiera**, gli **oggetti** sopra, e le **regole** del gioco.

### Costanti globali

| Nome | Valore | Significato |
|------|--------|-------------|
| `UP` | 0 | Muovi verso l'alto |
| `DOWN` | 1 | Muovi verso il basso |
| `LEFT` | 2 | Muovi verso sinistra |
| `RIGHT` | 3 | Muovi verso destra |

Sono solo dei nomi facili per non dover ricordare numeri a memoria.

### `SceneObject` — Un oggetto sulla scacchiera

```
SceneObject(label="chair", x=3, y=5, color=(0.1, 0.7, 0.2))
```

Contiene:
- `label`: il nome dell'oggetto (`"chair"`, `"table"`, `"lamp"`)
- `x`, `y`: la posizione sulla griglia (da 0 a 6)
- `color`: il colore RGB (serve solo per disegnare)

### `GridWorldEnv` — IL MONDO

È la classe principale. Gestisce tutto: la posizione del robottino, gli oggetti, i movimenti.

#### `__init__(size=7, seed=7, detector=None)`

Prepara un mondo nuovo. Cosa fa:
1. Crea una griglia `size × size` (7 per 7 = 49 caselle)
2. Prepara un generatore di numeri casuali con `seed` (se usi lo stesso seed, ottieni sempre le stesse posizioni — utile per testare)
3. Attacca un "detector" (gli occhi del robottino), oppure ne crea uno nuovo
4. Segna che l'oggetto da raggiungere è la sedia (`target_label = "chair"`)
5. Mette il robottino in posizione (0, 0) all'inizio
6. Prepara una lista vuota di oggetti

#### `reset()` — Ricomincia una partita

Cosa fa passo-passo:
1. Chiama `_sample_positions(4)` per trovare 4 posizioni che non si sovrappongono
2. Mette il robottino nella prima posizione
3. Crea 3 oggetti:
   - La **sedia** (colore verde) — il bersaglio
   - Il **tavolo** (colore blu) — distrattore/ostacolo
   - La **lampada** (colore giallo) — distrattore/ostacolo
4. Restituisce lo stato iniziale (cosa vede il robottino)

#### `step(action)` — Muovi di un passo

Questa è la funzione **più importante**. Riceve un'azione (0=su, 1=giù, 2=sinistra, 3=destra) e:

1. **Salva com'era la situazione prima** (posizione del robottino, distanza dalla sedia)
2. **Calcola la nuova posizione** senza muovere ancora il robottino
3. **Controlla se c'è un ostacolo** nella nuova posizione:
   - Se la casella contiene un tavolo o una lampada → **collisione!**
   - Il robottino **non si muove** (resta fermo)
   - Riceve una punizione di **-1.0** (tanto quanto la ricompensa per aver raggiunto la sedia)
   - Non ha finito (`done = False`), la partita continua
4. **Se non c'è collisione**, muove il robottino e calcola:
   - `done`: "ha raggiunto la sedia?" (confronta le coordinate)
   - Ricompensa base: **-0.01** per ogni passo (così impara a non perdere tempo)
   - Bonus se si avvicina alla sedia: `+0.02 * (distanza_prima - distanza_dopo)`. Se si allontana, diventa negativo!
   - Se resta fermo (caso raro, quando prova a uscire dalla griglia): **-0.03**
   - Se raggiunge la sedia: **+1.0**
5. Restituisce: `(osservazione, ricompensa, finito?, info)` — esattamente come richiesto dagli standard di RL

#### `_observation()` — Cosa vede il robottino (lo STATO)

Questa funzione costruisce un "numero di telefono" a 9 cifre. Ogni cifra è un numero intero.

```
(agent_x, agent_y, target_dx, target_dy, visible, danger_up, danger_down, danger_left, danger_right)
```

Spieghiamo ogni cifra:

| Cifra | Valori possibili | Cosa significa |
|-------|-----------------|----------------|
| `agent_x` | 0-6 | Posizione X del robottino |
| `agent_y` | 0-6 | Posizione Y del robottino |
| `target_dx` | -1, 0, 1 | Direzione della sedia sull'asse X (-1 = sinistra, 0 = stessa riga, 1 = destra) |
| `target_dy` | -1, 0, 1 | Direzione della sedia sull'asse Y (-1 = sotto, 0 = stessa colonna, 1 = sopra) |
| `visible` | 0 o 1 | La sedia è nel raggio di visione? (1 = sì) |
| `danger_up` | 0 o 1 | C'è un ostacolo nella casella SOPRA? (1 = attenzione!) |
| `danger_down` | 0 o 1 | C'è un ostacolo nella casella SOTTO? |
| `danger_left` | 0 o 1 | C'è un ostacolo nella casella SINISTRA? |
| `danger_right` | 0 o 1 | C'è un ostacolo nella casella DESTRA? |

Esempio: `(3, 2, 1, 1, 1, 0, 1, 0, 0)` significa:
> "Sono in posizione (3,2), la sedia è in basso a destra e la vedo,
> c'è un ostacolo sotto di me, attenzione a non andare giù!"

I 4 flag di **danger** sono la chiave per evitare gli ostacoli:
se `danger_left = 1`, il robottino sa che a sinistra c'è un tavolo o una lampada
e impara a **non andare a sinistra** (altrimenti prende -1.0).

#### `_target_object()` — Cerca la sedia

Scorre la lista degli oggetti e restituisce quello con etichetta `"chair"`.
Se non lo trova (dovrebbe essere impossibile), va in panico.

#### `_sample_positions(count)` — Pesca posizioni casuali

Genera `count` posizioni (coppie x,y) che non si sovrappongono.
Usa il generatore casuale interno (`self.random`), così con lo stesso `seed`
si ottengono sempre le stesse posizioni.

#### `_manhattan_distance(x1,y1, x2,y2)` — Distanza "a passi"

Calcola la distanza tra due caselle contando quanti passi servono per andare da una all'altra
muovendosi solo su/giù/sinistra/destra (come un taxi che si muove tra i palazzi).

Formula magica: `|x1 - x2| + |y1 - y2|` (dove `|` significa "senza segno")

---

## `detector.py` — Gli occhi del robottino

Questo modulo simula un **rilevatore di oggetti** (nella realtà sarebbe un'intelligenza artificiale
che guarda le immagini e dice "ecco una sedia!").

### `Detection` — Il risultato di una "guardata"

```
Detection(label="chair", dx=1, dy=1, visible=True)
```

Un pacchettino con:
- `label`: cosa ho visto
- `dx`, `dy`: direzione approssimativa verso l'oggetto
- `visible`: l'ho visto davvero o non si vede?

### `SemanticDetector` — Il rilevatore finto

#### `__init__(vision_radius=3)`

Prepara il rilevatore. Il `vision_radius` dice quanto lontano vede il robottino (3 caselle = vede mezza griglia).

#### `detect(objects, agent_position, target_label)`

Il cuore del detector. Prende la lista degli oggetti, la posizione del robottino,
e il nome dell'oggetto da cercare.

Cosa fa:
1. Cerca l'oggetto con l'etichetta giusta (es. `"chair"`)
2. Calcola la differenza di posizione: `dx = sedia.x - robot.x`, `dy = sedia.y - robot.y`
3. Controlla se la sedia è **visibile** (distanza ≤ `vision_radius` in entrambi gli assi)
4. **Se non è visibile**: restituisce `(dx=0, dy=0, visible=False)` — dice "non la vedo"
5. **Se è visibile**: restituisce la **direzione "arrotondata"** usando `_sign()`:
   - `dx` positivo → 1 (destra), negativo → -1 (sinistra), zero → 0
   - `dy` positivo → 1 (sopra), negativo → -1 (sotto), zero → 0
   - Questo trasforma "la sedia è a 3 caselle di distanza a destra" in un semplice "è a destra"

**Perché arrotondare?** Perché lo stato deve essere piccolo.
"Destra o sinistra" è abbastanza per imparare, non serve sapere esattamente a quanti passi.

#### `_sign(value)` — Arrotonda la direzione

| Valore | Risultato |
|--------|-----------|
| Negativo | -1 |
| Zero | 0 |
| Positivo | 1 |

Semplice!

---

## `agent.py` — Il cervello del robottino

Questo modulo implementa **Q-learning**, un algoritmo di apprendimento famoso.
Immagina una **tabella** (Q-table) con:
- **Righe**: tutti i possibili "stati" (cioè il numero di telefono a 9 cifre)
- **Colonne**: le 4 azioni (su, giù, sinistra, destra)
- **Celle**: un voto (Q-value) che dice "in questo stato, quest'azione è buona?"

### `State` — Il tipo dello stato

`tuple[int, ...]` — una sequenza di numeri interi di qualsiasi lunghezza.

### `QLearningAgent` — L'agente che impara

#### `__init__(actions=4, alpha=0.2, gamma=0.95, epsilon=0.5)`

Prepara il robottino:
- `actions`: 4 mosse possibili (su/giù/sinistra/destra)
- `alpha` (0.2): "velocità di apprendimento" — quanto velocemente impara dai nuovi dati
- `gamma` (0.95): "sconto sul futuro" — quanto conta una ricompensa futura rispetto a una immediata
- `epsilon` (0.5): "voglia di esplorare" — all'inizio il 50% delle mosse sono **casuali** (per scoprire il mondo)

La Q-table è un `defaultdict`: se chiedi uno stato mai visto, restituisce `[0.0, 0.0, 0.0, 0.0]`
(tutte le azioni sembrano ugualmente buone).

#### `choose_action(state)` — Decidi la prossima mossa

Il robottino lancia una monetina:
- Se esce **testa** (probabilità = `epsilon`): sceglie un'azione **a caso** (esplora!)
- Se esce **croce**: guarda nella Q-table e sceglie l'azione con il **voto più alto** (sfrutta!)

Dopo molti episodi, `epsilon` diventa piccolo e il robottino smette di esplorare,
usando solo quello che ha imparato.

#### `learn(state, action, reward, next_state, done)` — Impara dall'esperienza

Questa è la **formula magica del Q-learning**:

```
Q(stato, azione) += alpha * (ricompensa + gamma * max(Q(stato_nuovo)) - Q(stato, azione))
```

Scomponiamo:
1. **`Q(stato, azione)`**: il voto corrente per "in questo stato, fare questa azione"
2. **`ricompensa`**: cosa ho ottenuto? (-1.0 se ostacolo, +1.0 se sedia, ecc.)
3. **`gamma * max(Q(stato_nuovo))`**: "la luce in fondo al tunnel" — il voto migliore del nuovo stato (cosa mi aspetto di guadagnare in futuro)
4. **`ricompensa + gamma * max(Q(stato_nuovo))`**: il "target" — la vera ricompensa totale che mi aspetto
5. **`target - Q(stato, azione)`**: l'errore — quanto ho sbagliato la previsione?
6. **`alpha * errore`**: quanto voglio correggere il voto (un pezzettino alla volta)

Se `done = True` (partita finita), il futuro non esiste, quindi `max(Q(stato_nuovo)) = 0`.

#### `decay_exploration(minimum=0.05, factor=0.997)` — Diventa più saggio col tempo

Ad ogni episodio, riduce un po' la voglia di esplorare:

```
epsilon = max(0.05, epsilon × 0.997)
```

Dopo ~800 episodi, `epsilon` arriva a 0.05 = 5%. Il robottino è diventato "adulto"
e usa quasi sempre quello che ha imparato.

---

## `render.py` — Il pittore (OpenGL)

Prende il mondo (`GridWorldEnv`) e lo disegna in una finestra usando OpenGL.
Tutto è fatto in modo semplice ("modalità immediata"), senza shader o motori grafici.

### `RendererConfig` — Impostazioni grafiche

Piccole regolazioni su finestra e caratteri.

### `OpenGLRenderer` — Il disegnatore

#### `__init__(env, config)`

Riceve il mondo da disegnare e le impostazioni.

#### `initialize()` — Apre la finestra

1. Avvia GLFW (il sistema che gestisce le finestre)
2. Crea una finestra 720×720 pixel con titolo "Virtual World Explorer"
3. Imposta la proiezione 2D che va da (0,0) a (7,7) — come la griglia!
4. Sfondo grigio scuro

#### `draw()` — Dipinge un fotogramma

Chiamato a ogni passo. Cosa disegna (in ordine):
1. **Griglia**: linee orizzontali e verticali per separare le 49 caselle
2. **Oggetti**: quadrati colorati:
   - Sedia: **verde**
   - Tavolo: **blu**
   - Lampada: **gialla**
3. **robottino**: quadrato bianco leggermente più piccolo
4. **HUD**: un pannello in alto a sinistra con:
   - `TARGET: CHAIR` — cosa deve raggiungere
   - `AGENT: x, y` — posizione corrente
   - `VISIBLE: True/False` — la sedia è visibile?

#### `should_close()` — Devo chiudere?

Controlla se l'utente ha cliccato sulla X della finestra.

#### `poll()` — Ascolta gli eventi

Permette alla finestra di rispondere al mouse/tastiera.

#### `shutdown()` — Chiude tutto

Pulisce e libera la memoria.

#### `_draw_grid()` — Le linee della griglia

Disegna `size+1` linee orizzontali e verticali in grigio scuro.

#### `_draw_objects()` — I quadratini colorati

Ogni oggetto (sedia, tavolo, lampada) viene disegnato come un quadrato
con un po' di padding (rientro) per non toccare i bordi della casella.

#### `_draw_agent()` — Il robottino bianco

Un quadrato bianco, con un padding leggermente maggiore per distinguerlo dagli oggetti.

#### `_draw_hud_overlay(lines)` — Il pannello informativo

1. Cambia la proiezione per lavorare in pixel (invece che in coordinate di griglia)
2. Disegna un rettangolo scuro come sfondo del pannello
3. Per ogni riga di testo, chiama `_draw_text()` per scrivere le lettere
4. Ripristina la proiezione normale per il prossimo fotogramma

#### `_draw_text(x, y, text, scale)` — Disegna testo con pixel art

Per ogni carattere del testo:
1. Cerca il glifo (la "mappa" del carattere) nel dizionario `_GLYPHS`
2. Chiama `_draw_glyph()` per disegnarlo
3. Sposta il cursore a destra per il prossimo carattere

#### `_draw_glyph(x, y, glyph, scale)` — Un singolo carattere

Un carattere è una griglia 5×7 di bit (5 colonne, 7 righe).
Ogni riga è una stringa come `"01110"` dove `1` significa "accendi questo pixel".
Disegna tanti quadratini piccoli per formare la lettera.

### `_GLYPHS` — Il dizionario dei caratteri

Contiene le "mappe" di lettere e numeri: A, B, C, D, E, G, I, L, N, O, P, R, S, T, U, V, Y,
i segni `:` e `,`, e le cifre 0-9. Ogni carattere è una tupla di 7 stringhe (una per riga).

Esempio: la lettera **A**:

```
01110     ..##.
10001     #...#
10001     #...#
11111     #####
10001     #...#
10001     #...#
10001     #...#
```

---

## `main.py` — Il regista

Coordina tutto: addestra il robottino, poi mostra il risultato.

### `train_agent(episodes=5000, max_steps=50)`

Il ciclo di addestramento:

1. Crea il mondo (`GridWorldEnv`) e il robottino (`QLearningAgent`)
2. Per ogni episodio (5000 volte):
   - Ricomincia il mondo con `reset()` (nuove posizioni casuali)
   - Per massimo 50 passi:
     - Il robottino sceglie un'azione con `choose_action()`
     - Il mondo esegue l'azione con `step(action)` (si muove o sbatte)
     - Il robottino impara dall'esperienza con `learn()`
     - Se ha raggiunto la sedia, smette
   - Riduce l'esplorazione con `decay_exploration()`
   - Ogni 50 episodi stampa il progresso
3. Restituisce il mondo e il robottino addestrato

### `run_demo(env, agent, steps=None)`

Mostra il robottino in azione:
1. Crea il renderer (la finestra OpenGL)
2. **Congela l'esplorazione** (`epsilon = 0.0`): il robottino userà solo ciò che ha imparato
3. Per ogni passo:
   - Usa `_choose_action_without_loop()` per scegliere l'azione (evita i loop)
   - Muove il robottino e disegna la scena
   - Aspetta 0.35 secondi (così si vede il movimento)
   - Se raggiunge la sedia, resetta il mondo
4. Quando si chiude la finestra, ripristina `epsilon` e pulisce il renderer

### `_preview_position(env, action)` — Dove finirei?

Calcola dove andrebbe il robottino se eseguisse quell'azione.
Se la casella contiene un ostacolo, restituisce la posizione **corrente** (perché
il movimento viene bloccato da `step()`).

### `_choose_action_without_loop(env, state, agent, recent_positions)`

Una versione "intelligente" di `choose_action` che evita i loop:
1. Prende i voti della Q-table per lo stato corrente
2. Ordina le azioni dal voto più alto al più basso
3. Per ogni azione, calcola un punteggio:
   - Base = voto dalla Q-table
   - Se l'azione mi fa restare fermo: **-1.0**
   - Se l'azione porta in una casella visitata di recente: **-0.7**
   - Se l'azione mi fa tornare indietro (dov'ero un passo fa): **-0.5**
4. Sceglie l'azione col punteggio più alto

Questo sistema anti-loop è fondamentale: senza di esso, il robottino potrebbe
rimanere intrappolato a fare avanti e indietro (destra → sinistra → destra → sinistra → ...).

### `main()` — Via!

Unisce tutto:
1. Addestra il robottino per 5000 episodi
2. Mostra il risultato nella finestra OpenGL

---

## Il flusso completo (tutto insieme)

```
main()
  │
  ├─ train_agent(5000)
  │     │
  │     ├─ Crea: GridWorldEnv + QLearningAgent
  │     │
  │     └─ Per 5000 episodi:
  │           │
  │           ├─ env.reset()
  │           │     ├─ _sample_positions(4) → 4 caselle senza sovrapposizioni
  │           │     ├─ Mette il robottino in posizione
  │           │     ├─ Crea sedia + tavolo + lampada
  │           │     └─ _observation() → stato a 9 cifre
  │           │
  │           └─ Finché non raggiunge la sedia (max 50 passi):
  │                 │
  │                 ├─ agent.choose_action(state)
  │                 │     ├─ Se epsilon: azione casuale
  │                 │     └─ Se no: max dalla Q-table
  │                 │
  │                 ├─ env.step(action)
  │                 │     ├─ Calcola nuova posizione
  │                 │     ├─ C'è un ostacolo? → -1.0, resta fermo
  │                 │     ├─ Sedia raggiunta? → +1.0, finito
  │                 │     ├─ Altrimenti → -0.01 + bonus di avvicinamento
  │                 │     └─ _observation() → nuovo stato
  │                 │
  │                 └─ agent.learn(state, action, reward, next_state, done)
  │                       └─ Q(s,a) += alpha * (reward + gamma * max(Q(s')) - Q(s,a))
  │
  └─ run_demo(env, agent)
        │
        ├─ epsilon = 0.0 (niente esplorazione)
        │
        └─ Finché non chiudo la finestra:
              │
              ├─ _choose_action_without_loop()
              │     ├─ Prende i voti dalla Q-table
              │     ├─ Penalizza loop, posizioni recenti, movimenti inutili
              │     └─ Sceglie la migliore
              │
              ├─ env.step(action)
              │
              ├─ renderer.draw()
              │     ├─ _draw_grid()
              │     ├─ _draw_objects()
              │     ├─ _draw_agent()
              │     └─ _draw_hud_overlay()
              │
              ├─ time.sleep(0.35)  ← aspetta un po' per far vedere il movimento
              │
              └─ Se sedia raggiunta: env.reset() → ricomincia da capo
```

---

## Perché funziona?

### La Q-table

Immagina una **tabella** con migliaia di righe (gli stati possibili) e 4 colonne (le azioni).
All'inizio tutti i voti sono 0. Il robottino fa cose a caso.

Dopo aver provato:
- "Su → sedia!" → voto di SU aumenta (perché ha avuto +1.0)
- "Giù → ostacolo!" → voto di GIÙ diminuisce (perché ha avuto -1.0)
- "Su → mi sono avvicinato" → voto di SU aumenta un pochino

Dopo centinaia di episodi, la tabella contiene la **saggezza** del robottino:
per ogni situazione, sa qual è la mossa migliore.

### Le danger flag

Il trucco più importante: lo stato include 4 numeri che dicono
"c'è un ostacolo in questa direzione?". Così:

- Stato = `(3, 4, 0, 1, 1, 0, 0, 1, 0)` + azione = SINISTRA = collisione!
- `Q([3,4,0,1,1,0,0,1,0], SINISTRA)` diventa negativo
- robottino impara: da questo stato, non andare a sinistra!

Se la danger flag non ci fosse, il robottino non saprebbe **perché** ha preso -1.0
quando è andato a sinistra. Con la danger flag, impara ad associare
"danger_left = 1" con "non andare a sinistra".

### La penalità forte (-1.0)

Collidere con un ostacolo fa **-1.0**, esattamente quanto prendere la sedia fa **+1.0**.
Questo significa che anche un solo scontro annulla completamente la gioia di aver raggiunto
il bersaglio. Il robottino impara che **non vale mai la pena** sbattere.

---

## Cosa significa ogni numero?

### Lo stato (osservazione a 9 cifre)

```
(ax, ay, tdx, tdy, vis, dup, ddown, dleft, dright)
 ^    ^    ^     ^     ^    ^     ^      ^       ^
 1    2    3     4     5    6     7      8       9
```

| Posizione | Variabile | Esempio | Cosa dice |
|-----------|-----------|---------|-----------|
| 1 | `ax` | 3 | robottino è alla colonna 3 |
| 2 | `ay` | 2 | robottino è alla riga 2 |
| 3 | `tdx` | 1 | La sedia è a destra |
| 4 | `tdy` | -1 | La sedia è sotto |
| 5 | `vis` | 1 | La sedia è visibile |
| 6 | `dup` | 0 | Sopra è libero |
| 7 | `ddown` | 1 | **Sotto c'è un ostacolo!** |
| 8 | `dleft` | 0 | A sinistra è libero |
| 9 | `dright` | 0 | A destra è libero |

### La ricompensa

| Situazione | Ricompensa | Perché? |
|------------|-----------|---------|
| Passo normale | -0.01 | Ogni passo costa un po' per non perdere tempo |
| Mi avvicino alla sedia | +0.02 × (distanza_prima - distanza_dopo) | Insegnamento graduale |
| Mi allontano | Idem ma negativo | Impara che sta sbagliando strada |
| Resto fermo (fuori dalla griglia) | -0.03 (aggiunto al normale) | "Non fare lo gnocco" |
| **Collisione con ostacolo** | **-1.0** | **Punizione grossa** |
| **Sedia raggiunta** | **+1.0** | **Grande ricompensa!** |

---

## Perché queste scelte tecniche?

| Scelta | Motivo |
|--------|--------|
| Q-tabella invece di rete neurale | Più semplice, trasparente, non serve GPU |
| Coordinate assolute (0-6) invece che one-hot | Stato compatto (~14k invece di milioni) |
| Danger flag invece di coordinate degli ostacoli | Stato ancora più piccolo, focus sull'informazione utile |
| Penalità -1.0 per collisione | Pari alla ricompensa del target: ogni collisione è inaccettabile |
| Vision radius per il detector | Simula occhi reali (non vede tutta la griglia) |
| Episodi con seed | Riproducibile — stessi semi = stessi risultati |
| OpenGL senza shader | Minimo indispensabile per visualizzare |

---

## Riferimenti

- **Q-learning**: Sutton & Barto, "Reinforcement Learning: An Introduction"
- **OpenGL**: A. Shreiner et al., "OpenGL Programming Guide" (la "Red Book")
- **GLFW**: https://www.glfw.org/ — documentazione ufficiale
