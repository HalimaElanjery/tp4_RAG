
package ht.fds.mbds.elanjery.tp4_rag_halima;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;

import dev.langchain4j.memory.chat.MessageWindowChatMemory;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.model.input.PromptTemplate;

import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;

import dev.langchain4j.service.AiServices;

import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class Test3 {

    public static void main(String[] args) {

        // PHASE 1 : Création du modèle d'embedding

        EmbeddingModel embeddingModel =
                new AllMiniLmL6V2EmbeddingModel();

        // PHASE 2 : Chargement et indexation du document

        InMemoryEmbeddingStore<TextSegment> embeddingStore =
                creerEmbeddingStore("rag.pdf", embeddingModel);

        // PHASE 3 : Création du ContentRetriever

        ContentRetriever contentRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(embeddingStore)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        System.out.println("ContentRetriever créé.");

        // PHASE 4 : Initialisation de Gemini

        String apiKey = System.getenv("GEMINI_KEY");

        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException(
                    "La variable d'environnement GEMINI_KEY est absente."
            );
        }

        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-3.5-flash-lite")
                .temperature(0.7)
                .logRequests(true)
                .logResponses(true)
                .build();

        System.out.println("Gemini initialisé avec logging.");

        // PHASE 5 : Création du PromptTemplate

        PromptTemplate promptTemplate = PromptTemplate.from("""
                Est-ce que la requête '{{question}}' porte sur le 'RAG'
                (Retrieval Augmented Generation) ou le 'Fine Tuning' ?
                Réponds seulement par 'oui', 'non', ou 'peut-être'.
                """);

        System.out.println("PromptTemplate créé.");

        // PHASE 6 : Création du QueryRouter personnalisé

        QueryRouter queryRouter = new QueryRouter() {

            @Override
            public Collection<ContentRetriever> route(Query query) {

                String prompt = promptTemplate
                        .apply(Map.of("question", query.text()))
                        .text();

                System.out.println("\nQuestion reçue : " + query.text());
                System.out.println("Prompt envoyé à Gemini :");
                System.out.println(prompt);

                String reponse = model.chat(prompt);

                System.out.println("Réponse de Gemini : " + reponse);

                if (reponse.toLowerCase().contains("non")) {

                    System.out.println(
                            "Décision : aucun document ne sera consulté."
                    );

                    return Collections.emptyList();

                } else {

                    System.out.println(
                            "Décision : consultation du document RAG."
                    );

                    return Collections.singletonList(contentRetriever);
                }
            }
        };

        System.out.println("QueryRouter personnalisé créé.");

        // PHASE 7 : Création du RetrievalAugmentor

        RetrievalAugmentor retrievalAugmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryRouter(queryRouter)
                        .build();

        System.out.println("RetrievalAugmentor créé.");

        // PHASE 8 : Création de l'assistant

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        System.out.println("Assistant créé.");

        // PHASE 9 : Boucle de conversation

        Scanner scanner = new Scanner(System.in);

        System.out.println("\n===== TEST 4 : ROUTAGE SANS RAG =====");
        System.out.println("Saisissez 'exit' pour quitter.");

        while (true) {

            System.out.print("\nPose ta question : ");

            String question = scanner.nextLine();

            if (question.equalsIgnoreCase("exit")) {
                break;
            }

            String reponse = assistant.chat(question);

            System.out.println("\nRéponse de l'assistant :");
            System.out.println(reponse);
        }

        scanner.close();

        System.out.println("\nFin du programme.");
    }

    private static InMemoryEmbeddingStore<TextSegment> creerEmbeddingStore(
            String nomFichier,
            EmbeddingModel embeddingModel) {

        ApacheTikaDocumentParser parser =
                new ApacheTikaDocumentParser();

        Document document =
                ClassPathDocumentLoader.loadDocument(nomFichier, parser);

        System.out.println("Document chargé : " + nomFichier);

        DocumentSplitter splitter =
                DocumentSplitters.recursive(300, 30);

        List<TextSegment> segments =
                splitter.split(document);

        System.out.println("Nombre de segments : " + segments.size());

        List<Embedding> embeddings =
                embeddingModel.embedAll(segments).content();

        System.out.println("Nombre d'embeddings : " + embeddings.size());

        InMemoryEmbeddingStore<TextSegment> embeddingStore =
                new InMemoryEmbeddingStore<>();

        embeddingStore.addAll(embeddings, segments);

        System.out.println("Stockage vectoriel terminé.");

        return embeddingStore;
    }
}
