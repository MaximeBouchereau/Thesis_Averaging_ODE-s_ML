=== ENGLISH VERSION BELOW ===

Resolution approchée d'équations différentielles fortement oscillantes via l'utilisation du Machine Learning

### Averaging_ML ###

Code Python permettant l'approximation de solutions d'équations différentielles de la forme $$y'(t) = f\left(t/\epsilon,y(t)\right)$$ où $f$ est une fonction périodique en sa première variable. Stratégie: décomposition "slow-fast", composition d'un flot associé à un champ moyenné (dynamique lente) avec un générateur de fortes oscillations (dynamique rapide).

Approximation du champ moyenné: dynamique lente, EDO autonome, analyse rétrograde (problème étudié dans le cas autonome)
Approximation du générateur de fortes oscillations: dynamique rapide, auto-encodeur.

Etapes suivies:

- Création de données
- Entraînement des réseaux de neurones
- Intégration numérique: décomposition "slow-fast" et schéma Micro-Macro avec les champs appris.

Possibilités avec les fonctions du programme:

- Création de données
- Apprentissage via minimisation de Loss
- Intégration numérique slow-fast & Micro-Macro
- Courbes de convergence: erreur numérique (méthode slow-fast ou Micro-Macro)
- Courbes de convergence: test du caractère UA de la méthode Micro-Macro avec les champs appris.
- Courbes de convergence: approximation du champ moyenné modifié ainsi que du générateur de fortes oscillations, erreurs par rapport à $\epsilon$.

### Averaging_Autonomous_ML ###

Code Python permettant l'approximation de solutions d'équations différentielles de la forme $$y' = (1/\epsilon)Ay + g(y)$$, où $A$ est une matrice donc les valeurs prores sont des multiples de $i$. Avec un changement de variable (et donc de fonction), il est possible de se ramener au cas précédent. Mais une nouvelle approche est testée, en regardant ici la solution comme composition de deux flots (un pour la dynamique lente, l'autre pour la dynamique rapide). 

Apprentissage des deux flots via des réseaux de neurones "classiques" (pas d'auto-encodeur cette fois).

Etapes suivies:

- Création de données
- Entraînement des réseaux de neurones
- Intégration numérique: méthode numérique.

Possibilités avec les fonctions du programme:

- Création de données
- Apprentissage via minimisation de Loss
- Intégration numérique via une méthode numérique choisie
- Courbes de convergence: erreur numérique (méthode numérique)

Une comparaison entre les deu programmes peut-être réalisée.







=== ENGLISH VERSION HERE ===

Approximation of solutions of highly oscillatory equations with Machine Learning

### Averaging_ML ###

Python code for approximation of solutions of differential equations of the form $$y'(t) = f\left(t/\epsilon,y(t)\right)$$ where $f$ is periodic w.r.t. its first variable. Strategy: slow-fast decomposition, slow dynamics (associated to averaged field) and fast dynamics (associated to high oscillation generator).

Approximation of averaged field: slow dynamics, autonomous ODE, backward error analysis (case of autonomous ODE's and ML)
Approximation of high oscillation generator: fast dynamics, autoencoder.

Steps:

- Data creation
- Training of neural networks
- Numerical integration: slow-fast and Micro-Macro scheme with approximated fields.

Possibilities with functions of the code:

- Data creation
- Learning via Loss minimization
- Numerical integration: slow-fast & Micro-Macro
- Convergence curves: numerical error (slow-fast or Micro-Macro)
- Convergence curves: test of UA property for Micro-Macro method with approximated fields.
- Convergence curves: approximation of modified averaged field and high oscillation generator, errors w.r.t. $\epsilon$.

### Averaging_Autonomous_ML ###

Python code for numerical approximation of solutions of differential equations of the form $$y' = (1/\epsilon)Ay + g(y)$$, where $A$ is a matrix whose eigenvalues are multiples of $i$. With variable (and function) change, one can get the previous form. Here, a new approach is tested. Solution is considered as composition of two flows (a first flow for slow dynamics and a second flow for fast dynamics).

Learning of bothn flows with MLP's (no autoencoder is required here).

Steps:

- Data creation
- Training of neural networks
- Numerical integration: numerical method

Possibilities with functions of the code:

- Data creation
- Learning via Loss minimization
- Numerical integration with a selected numerical method
- Convergence curves: numerical error

A comparison between both codes is possible.

