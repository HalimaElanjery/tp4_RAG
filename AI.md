# IA.md — Utilisation de l'intelligence artificielle dans le TP4 RAG

## Introduction

Dans le cadre du TP4 consacré au RAG avec LangChain4j, j'ai utilisé ChatGPT comme outil d'assistance lorsque je rencontrais des difficultés techniques ou lorsque je ne comprenais pas certains concepts.

Je ne me suis pas limitée à demander du code. J'ai également demandé des explications sur le fonctionnement des composants, les erreurs rencontrées et les choix techniques effectués.

Les échanges avec l'IA m'ont permis de comprendre progressivement les différentes étapes de construction d'un système RAG, depuis le chargement des documents jusqu'au routage intelligent des sources de récupération.

---

# 1. Configuration du projet et dépendances Maven

### Problème rencontré

Au début du TP, j'ai rencontré plusieurs problèmes de configuration Maven.

IntelliJ IDEA affichait notamment des erreurs telles que :

- `Cannot resolve symbol 'dev'`
- `Cannot resolve symbol 'ChatModel'`
- `Cannot resolve symbol 'Document'`
- `Cannot resolve symbol 'TextSegment'`

Certaines dépendances LangChain4j n'étaient pas correctement résolues. J'avais également une difficulté avec la version du BOM et le choix de la dépendance du modèle d'embeddings.

### Raison de l'utilisation de l'IA

Je ne comprenais pas pourquoi les classes de LangChain4j étaient introuvables alors que les imports étaient présents dans mon code.

J'ai utilisé l'IA pour identifier les problèmes de configuration Maven et comprendre le rôle des dépendances nécessaires au TP.

### Prompt utilisé

« Je réalise le TP4 RAG avec Java, Maven et LangChain4j. IntelliJ affiche Cannot resolve symbol dev, ChatModel, Document, DocumentSplitter et TextSegment. Voici mon pom.xml. Analyse les dépendances et indique les corrections nécessaires pour que le projet fonctionne. »

### Réponse de l'IA

L'IA m'a expliqué que les erreurs provenaient principalement de la configuration des dépendances.

Elle m'a notamment aidée à :
- Corriger la version du BOM LangChain4j.
- Utiliser les dépendances adaptées au modèle Gemini.
- Ajouter le parseur Apache Tika pour lire les documents PDF.
- Utiliser `langchain4j-embeddings-all-minilm-l6-v2` pour les embeddings.
- Ajouter les dépendances nécessaires au fonctionnement du RAG.

Elle m'a également expliqué qu'après une modification du `pom.xml`, il fallait recharger le projet Maven afin qu'IntelliJ télécharge et reconnaisse les bibliothèques.

### Ce que j'ai appris

J'ai compris que les erreurs d'importation ne signifient pas toujours que le code Java est incorrect.

Les classes utilisées dans un projet Maven doivent être fournies par des dépendances correctement déclarées et résolues.

J'ai également appris à distinguer un problème de dépendance Maven d'une erreur de programmation.

---

# 2. Test 1 — RAG naïf et construction du pipeline

### Problème rencontré

Je devais construire un système RAG à partir du fichier `rag.pdf`, mais je ne comprenais pas précisément les différentes étapes ni le rôle des classes de LangChain4j.

Je ne savais pas comment passer d'un document PDF à une recherche sémantique permettant de fournir des passages pertinents au modèle Gemini.

### Raison de l'utilisation de l'IA

J'ai demandé à l'IA de m'expliquer le fonctionnement du RAG et de m'accompagner dans la réalisation du premier test, étape par étape.

### Prompt utilisé

« Je dois réaliser le Test 1 du TP4 : RAG naïf. Explique-moi les étapes de réalisation avant de me donner le code. Je veux comprendre le rôle de chaque classe utilisée et avancer progressivement. »

### Réponse de l'IA

L'IA m'a expliqué le fonctionnement général du pipeline :

1. Charger le PDF avec `ClassPathDocumentLoader`.
2. Extraire son contenu avec Apache Tika.
3. Découper le document en segments avec `DocumentSplitters`.
4. Transformer les segments en vecteurs numériques à l'aide d'un modèle d'embeddings.
5. Enregistrer les vecteurs dans un `InMemoryEmbeddingStore`.
6. Utiliser `EmbeddingStoreContentRetriever` pour retrouver les passages pertinents.
7. Transmettre les passages récupérés au modèle Gemini avec la question de l'utilisateur.
8. Utiliser `MessageWindowChatMemory` pour conserver le contexte conversationnel.

L'IA m'a également expliqué la différence entre les composants :

- `Document` représente le document chargé.
- `TextSegment` représente une partie du document.
- `Embedding` représente la représentation vectorielle d'un segment.
- `EmbeddingStore` stocke les représentations vectorielles.
- `ContentRetriever` recherche les passages pertinents.
- `RetrievalAugmentor` enrichit la question avec les informations récupérées.

### Ce que j'ai appris

J'ai compris qu'un RAG ne consiste pas simplement à envoyer un PDF au modèle.

Le document doit d'abord être préparé, découpé et transformé en embeddings. Lorsqu'une question est posée, le système recherche les segments les plus proches sémantiquement de cette question.

J'ai également compris que le modèle de langage ne réalise pas lui-même la recherche vectorielle : cette recherche est effectuée par le retriever.

---

# 3. Test 2 — Transformation de la question

### Problème rencontré

Lors de mes tests, j'ai posé deux questions successives :

- Q1 : « Qu'est-ce que le fine-tuning ? »
- Q2 : « Dans le support de cours fourni, à quoi c'est comparé ? »

La deuxième question dépend de la première, car l'expression « c'est » fait référence au fine-tuning.

Sans transformation, la recherche documentaire utilisait principalement la deuxième question seule. Elle retrouvait alors des passages sur le reclassement, le routage et la recherche sémantique, qui ne répondaient pas directement à la question.

J'ai également constaté que la mémoire conversationnelle ne suffisait pas à rendre automatiquement la recherche documentaire contextuelle.

### Raison de l'utilisation de l'IA

Je ne comprenais pas pourquoi l'assistant pouvait conserver le contexte de la conversation tout en effectuant une recherche documentaire inadaptée.

J'ai demandé à l'IA d'expliquer cette différence et de m'aider à utiliser `CompressingQueryTransformer`.

### Prompt utilisé

« Dans le Test 2, je dois utiliser Query Transformation. J'ai posé "Qu'est-ce que le fine-tuning ?" puis "Dans le support de cours fourni, à quoi c'est comparé ?". Explique-moi pourquoi la deuxième recherche ne comprend pas correctement le contexte et comment utiliser CompressingQueryTransformer. »

### Réponse de l'IA

L'IA m'a expliqué qu'il faut distinguer deux mécanismes :

- La mémoire conversationnelle conserve les anciens messages pour permettre au modèle de répondre en tenant compte de l'historique.
- La transformation de question intervient avant la recherche documentaire afin de reformuler une question dépendante du contexte en une question autonome.

Elle m'a présenté `CompressingQueryTransformer`, qui utilise le contexte conversationnel pour produire une question plus complète avant de lancer la recherche vectorielle.

La question transformée permet alors de préciser le sujet du fine-tuning au lieu de conserver uniquement la référence « c'est ».

### Ce que j'ai appris

J'ai compris que la mémoire conversationnelle et la transformation de question ont des rôles différents.

La mémoire aide le modèle à comprendre la conversation, tandis que le Query Transformer améliore la requête utilisée pour rechercher les documents.

J'ai également compris qu'une question grammaticalement correcte peut être insuffisante pour une recherche sémantique si elle dépend d'une référence présente dans un message précédent.

---

# 4. Test 3 — Routage entre deux documents PDF

### Problème rencontré

Je devais utiliser deux documents PDF différents et faire en sorte que le système sélectionne le bon document selon la question.

Je ne savais pas comment associer plusieurs magasins d'embeddings à un mécanisme de routage.

### Raison de l'utilisation de l'IA

J'ai demandé à l'IA de m'expliquer le rôle de `LanguageModelQueryRouter`, la manière de définir les descriptions des sources et la façon de vérifier le choix effectué par le modèle.

### Prompt utilisé

« Je dois réaliser le routage entre deux PDF avec LangChain4j. Explique-moi comment créer un retriever pour chaque document, utiliser LanguageModelQueryRouter et vérifier dans les logs que le bon retriever est sélectionné. »

### Réponse de l'IA

L'IA m'a expliqué qu'il faut :

1. Charger chaque document séparément.
2. Créer un magasin d'embeddings pour chaque document.
3. Créer un `ContentRetriever` pour chaque magasin.
4. Associer chaque retriever à une description compréhensible.
5. Utiliser `LanguageModelQueryRouter` pour demander au modèle de choisir les sources adaptées à la question.
6. Configurer le `RetrievalAugmentor` avec le routeur.

Dans mon projet, les deux sources étaient :

- `rag.pdf` : support de cours sur le RAG, le fine-tuning, l'intelligence artificielle et les concepts associés.
- `reseaux-informatiques.pdf` : support sur les réseaux informatiques.

Le routeur a permis de sélectionner le document correspondant à la question posée.

### Ce que j'ai appris

J'ai compris que le routage permet d'éviter de rechercher dans tous les documents sans distinction.

J'ai également appris l'importance des descriptions des retrievers : elles donnent au modèle des informations sur le contenu de chaque source afin qu'il puisse effectuer un choix pertinent.

---

# 5. Test 4 — RAG conditionnel

### Problème rencontré

Je devais empêcher le système de lancer une recherche documentaire lorsqu'une question ne concerne pas le contenu du PDF.

Par exemple, pour une question comme « Bonjour », il n'est pas nécessaire de récupérer des passages du support de cours.

Je ne comprenais pas comment retourner une liste vide de retrievers lorsque le RAG n'est pas nécessaire.

### Raison de l'utilisation de l'IA

J'ai demandé à l'IA de m'expliquer comment créer un `QueryRouter` personnalisé et comment utiliser directement le modèle de langage pour prendre une décision avant la récupération documentaire.

### Prompt utilisé

« Je ne comprends pas comment réaliser le Test 4. Je dois créer un QueryRouter personnalisé qui décide si la question porte sur le RAG ou le Fine-Tuning. Explique-moi le fonctionnement du routeur, du PromptTemplate et de la liste des ContentRetriever. »

### Réponse de l'IA

L'IA m'a expliqué que le routeur peut interroger directement le modèle avec un prompt demandant de répondre par :

- `oui`
- `non`
- `peut-être`

Le prompt utilisé dans mon code était :

```text
Est-ce que la requête '{{question}}' porte sur le 'RAG'
(Retrieval Augmented Generation) ou le 'Fine Tuning' ?
Réponds seulement par 'oui', 'non', ou 'peut-être'.
```

Le code analyse ensuite la réponse du modèle :

- Si la réponse contient « non », le routeur retourne une liste vide avec `List.of()`.
- Sinon, il retourne le retriever du PDF.

### Ce que j'ai appris

J'ai compris qu'un `QueryRouter` ne sert pas uniquement à choisir entre plusieurs documents. Il peut également décider de ne consulter aucune source.

Une liste vide signifie qu'aucun retriever ne doit être utilisé pour cette requête.

J'ai aussi compris l'intérêt de séparer la décision de routage de la génération de la réponse finale.

---

# 6. Test 5 — RAG avec récupération sur le Web

### Problème rencontré

Je devais ajouter une deuxième source de récupération : le Web.

Je ne savais pas comment intégrer Tavily à LangChain4j ni comment combiner un retriever basé sur un PDF avec un retriever basé sur une recherche Web.

J'ai également rencontré une difficulté avec la variable d'environnement `TAVILY_API_KEY` : le programme indiquait qu'elle n'était pas configurée, alors que je l'avais déjà ajoutée dans Windows.

### Raison de l'utilisation de l'IA

J'ai demandé à l'IA de m'expliquer l'intégration de Tavily et de m'aider à résoudre le problème de variable d'environnement.

### Prompt utilisé

« Je dois ajouter Tavily au Test 5 du TP4. Explique-moi comment créer WebSearchEngine avec TavilyWebSearchEngine, comment créer WebSearchContentRetriever et comment utiliser DefaultQueryRouter avec le retriever PDF et le retriever Web. »

Pour le problème de la clé API :

« La variable TAVILY_API_KEY existe dans Windows, mais IntelliJ indique qu'elle n'est pas configurée. Comment résoudre ce problème ? »

### Réponse de l'IA

L'IA m'a expliqué que Tavily fournit un moteur de recherche Web utilisable avec LangChain4j.

Les principaux composants sont :

- `TavilyWebSearchEngine` : moteur de recherche Web.
- `WebSearchContentRetriever` : récupérateur qui transforme les résultats Web en contenus utilisables par le RAG.
- `DefaultQueryRouter` : routeur qui transmet la question aux retrievers configurés.

Pour le problème de la variable d'environnement, l'IA m'a expliqué qu'IntelliJ peut conserver les variables d'environnement disponibles au moment de son lancement.

La solution appliquée a été de redémarrer IntelliJ afin qu'il récupère la variable `TAVILY_API_KEY`.

### Ce que j'ai appris

J'ai compris qu'un système RAG peut utiliser différentes sources de connaissances, notamment des documents privés et des informations disponibles sur le Web.

J'ai également appris que les variables d'environnement sont utiles pour stocker les clés API sans les écrire directement dans le code source.

---

# 7. Bonus — Routage intelligent entre PDF et Web

### Problème rencontré

Après avoir intégré Tavily, j'ai utilisé `DefaultQueryRouter` avec deux retrievers :

- Le retriever PDF.
- Le retriever Web.

J'ai ensuite posé une question sur les élections américaines de mi-mandat de novembre 2026.

Les logs montraient que des passages du PDF étaient ajoutés au prompt, même si la question concernait uniquement une actualité.

Je ne comprenais pas pourquoi le système consultait le PDF alors que les informations recherchées étaient disponibles sur le Web.

### Raison de l'utilisation de l'IA

J'ai demandé à l'IA d'analyser les logs, d'expliquer le comportement de `DefaultQueryRouter` et de m'aider à améliorer le routage sans supprimer l'ancien code.

### Prompt utilisé

« Avec DefaultQueryRouter, les deux retrievers PDF et Web sont interrogés. Même pour une question d'actualité, des extraits du PDF sont ajoutés au prompt. Explique pourquoi et aide-moi à résoudre ce problème avec un routage intelligent qui choisit PDF, WEB ou BOTH. Je veux conserver l'ancien code en commentaire et avoir le fichier Test4.java complet. »

### Réponse de l'IA

L'IA m'a expliqué que `DefaultQueryRouter` transmet la requête à tous les retrievers configurés. Il ne réalise donc pas une sélection intelligente des sources.

Elle m'a proposé de remplacer ce comportement par un `QueryRouter` personnalisé utilisant Gemini pour classifier la question.

Les trois décisions possibles sont :

- `PDF` : sélectionner uniquement le retriever du document.
- `WEB` : sélectionner uniquement le retriever Web.
- `BOTH` : sélectionner les deux retrievers lorsque les deux sources sont nécessaires.

L'ancien bloc `DefaultQueryRouter` a été conservé en commentaire afin de montrer le fonctionnement initial et la modification apportée.

Pendant cette correction, j'ai également rencontré des erreurs de compilation :

**Erreur 1 : mauvais package pour QueryRouter**

J'avais utilisé :

```java
import dev.langchain4j.rag.router.QueryRouter;
```

L'import correct était :

```java
import dev.langchain4j.rag.query.router.QueryRouter;
```

**Erreur 2 : méthode aiMessage() introuvable**

J'avais utilisé :

```java
String decision = chatModel.chat(prompt)
        .aiMessage()
        .text()
        .trim()
        .toUpperCase();
```

IntelliJ affichait :

```text
Cannot resolve method 'aiMessage' in 'String'
```

L'IA m'a expliqué que, dans la version de LangChain4j utilisée dans mon projet, `chatModel.chat(String)` retourne directement une chaîne de caractères.

La correction était donc :

```java
String decision = chatModel.chat(prompt)
        .trim()
        .toUpperCase();
```

### Résultats des tests après correction

**Premier test :**

Question :

« Qu'est-ce que le fine-tuning ? »

Décision du routeur :

```text
PDF
```

Les logs montrent que seuls les passages du support de cours sont transmis au modèle. Aucun résultat Tavily n'est ajouté.

**Deuxième test :**

Question :

« Quels sont les résultats des élections américaines de mi-mandat de novembre 2026 ? »

Décision du routeur :

```text
WEB
```

Les logs montrent que les résultats de Tavily sont transmis au modèle sans nouvel extrait du PDF.

### Ce que j'ai appris

J'ai compris la différence entre un routage systématique et un routage intelligent.

`DefaultQueryRouter` transmet la requête à tous les retrievers configurés, tandis qu'un routeur personnalisé peut sélectionner uniquement les sources pertinentes.

J'ai également appris à lire les logs pour vérifier le résultat d'une décision de routage et à utiliser les messages d'erreur du compilateur pour identifier les problèmes de compatibilité avec l'API d'une bibliothèque.

---

# 8. Erreurs techniques supplémentaires et apprentissages

Pendant la réalisation du TP, j'ai rencontré plusieurs erreurs qui m'ont permis de mieux comprendre Java et LangChain4j.

### Problème 1 — Import incorrect de Embedding

J'avais utilisé un mauvais package pour la classe `Embedding`.

L'import correct était :

```java
import dev.langchain4j.data.embedding.Embedding;
```

J'ai appris qu'il faut vérifier le package réel d'une classe et ne pas confondre les modèles d'embeddings avec les objets représentant les vecteurs générés.

### Problème 2 — Avertissements Java et bibliothèques natives

Lors de l'exécution, j'ai observé des avertissements concernant l'accès natif d'ONNX Runtime et des bibliothèques utilisées par les tokenizers.

J'ai appris à distinguer les avertissements d'exécution des véritables erreurs qui empêchent le programme de compiler ou de fonctionner.

Un avertissement n'implique pas nécessairement que le programme a échoué.

### Problème 3 — Comprendre les logs

Les logs contenaient plusieurs informations : les requêtes envoyées à Gemini, les réponses du modèle, les passages récupérés et les décisions du routeur.

J'ai appris à les lire pour suivre le fonctionnement interne du système et identifier à quelle étape un problème se produit.

---

# 9. Bilan personnel

Cette réalisation m'a permis de mieux comprendre le fonctionnement d'un système RAG et les différentes possibilités offertes par LangChain4j.

J'ai particulièrement retenu les notions suivantes :

- La différence entre un document, un segment et un embedding.
- Le rôle d'un magasin vectoriel dans la recherche sémantique.
- La différence entre la mémoire conversationnelle et la transformation d'une question.
- Le fonctionnement d'un `ContentRetriever`.
- Le rôle d'un `RetrievalAugmentor`.
- La sélection des sources avec un `QueryRouter`.
- L'intégration d'une recherche Web avec Tavily.
- L'intérêt des logs pour comprendre et déboguer un système utilisant un modèle de langage.

J'ai également appris qu'obtenir une réponse générée par l'IA ne signifie pas nécessairement que le problème est compris ou correctement résolu.

Il est important de lire les explications, de comprendre les modifications proposées, de les tester et de vérifier les résultats.

L'objectif de cette utilisation de l'IA était donc de m'aider à surmonter les difficultés rencontrées tout en développant progressivement ma compréhension des concepts et des outils utilisés dans ce TP.
