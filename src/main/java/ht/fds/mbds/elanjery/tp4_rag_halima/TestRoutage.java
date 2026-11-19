
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

import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;

import dev.langchain4j.service.AiServices;

import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class TestRoutage {

    public static void main(String[] args) {

        // PHASE 1 : Ingestion des deux documents

        System.out.println("Création du modèle d'embedding...");
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        System.out.println("\nIngestion du support de cours sur l'IA...");
        InMemoryEmbeddingStore<TextSegment> storeIA =
                creerEmbeddingStore("rag.pdf", embeddingModel);

        System.out.println("\nIngestion du support sur les réseaux informatiques...");
        InMemoryEmbeddingStore<TextSegment> storeReseaux =
                creerEmbeddingStore("reseaux-informatiques.pdf", embeddingModel);

        // PHASE 2 : Création des deux ContentRetriever

        System.out.println("\nCréation des ContentRetriever...");

        ContentRetriever retrieverIA =
                creerContentRetriever(storeIA, embeddingModel);

        ContentRetriever retrieverReseaux =
                creerContentRetriever(storeReseaux, embeddingModel);

        // PHASE 3 : Initialisation de Gemini

        System.out.println("\nInitialisation de Gemini...");

        String apiKey = System.getenv("GEMINI_KEY");

        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException(
                    "La variable d'environnement GEMINI_KEY est absente."
            );
        }

        ChatModel chatModel = GoogleGenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-3.5-flash-lite")
                .temperature(0.7)
                .logRequests(true)
                .logResponses(true)
                .build();

        System.out.println("Gemini initialisé avec logging.");

        // PHASE 4 : Description des sources pour le routage

        Map<ContentRetriever, String> retrieverToDescription =
                new HashMap<>();

        retrieverToDescription.put(
                retrieverIA,
                "Support de cours portant sur l'intelligence artificielle, "
                        + "le machine learning, le deep learning et les concepts liés à l'IA."
        );

        retrieverToDescription.put(
                retrieverReseaux,
                "Document portant sur les réseaux informatiques, "
                        + "les modèles OSI et TCP/IP, l'adressage IP et les protocoles réseau."
        );

        // PHASE 5 : Création du QueryRouter

        System.out.println("\nCréation du LanguageModelQueryRouter...");

        QueryRouter queryRouter =
                new LanguageModelQueryRouter(chatModel, retrieverToDescription);

        // PHASE 6 : Création du RetrievalAugmentor

        System.out.println("\nCréation du RetrievalAugmentor...");

        RetrievalAugmentor retrievalAugmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryRouter(queryRouter)
                        .build();

        // PHASE 7 : Création de l'assistant

        MessageWindowChatMemory chatMemory =
                MessageWindowChatMemory.withMaxMessages(10);

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .chatMemory(chatMemory)
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        System.out.println("\nAssistant avec routage prêt !");

        // PHASE 8 : Conversation

        conversationAvec(assistant);
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

    private static ContentRetriever creerContentRetriever(
            InMemoryEmbeddingStore<TextSegment> embeddingStore,
            EmbeddingModel embeddingModel) {

        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();
    }

    private static void conversationAvec(Assistant assistant) {

        try (Scanner scanner = new Scanner(System.in)) {

            System.out.println("\n======================================");
            System.out.println("       ASSISTANT RAG AVEC ROUTAGE");
            System.out.println("======================================");
            System.out.println("Pose tes questions sur les documents.");
            System.out.println("Tape 'fin' pour quitter.");

            while (true) {

                System.out.println("\nPose ta question : ");
                String question = scanner.nextLine();

                if (question.isBlank()) {
                    continue;
                }

                if ("fin".equalsIgnoreCase(question)) {
                    System.out.println("Fin de la conversation.");
                    break;
                }

                String reponse = assistant.chat(question);

                System.out.println("\nAssistant : ");
                System.out.println(reponse);
            }
        }
    }
}
