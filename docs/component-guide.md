# Component Guide — Virtual World Explorer 3D

Questo documento spiega **ogni singola funzione** del progetto "Virtual World Explorer",
come si incastrano tra loro, e le ultime evoluzioni tecniche introdotte (il passaggio al 3D e alla gestione delle texture).

---

## Che cosa fa questo progetto?

Immagina un **robottino** (l'agente) che si muove su una **scacchiera 7×7**.
Sulla scacchiera ci sono tre oggetti tridimensionali o testurizzati: una **sedia** (il bersaglio da raggiungere),
un **tavolo** e una **lampada** (distrattori / ostacoli).

Il robottino all'inizio non sa dove sia la sedia. Deve **imparare da solo**,
provando e sbagliando. Ogni volta che si avvicina alla sedia, riceve una **ricompensa positiva** (caramella).
Ogni volta che sbatte contro un ostacolo, riceve una **punizione** e resta fermo.
Alla fine, dopo tanti tentativi, impara a raggiungere la sedia **evitando** tavolo e lampada.

Il tutto è mostrato in una **finestra grafica 3D (OpenGL)** inclinata prospetticamente, dove gli oggetti sono renderizzati come cartelli scontrornati (Billboard) caricati da immagini reali.

---

## Come sono organizzati i file