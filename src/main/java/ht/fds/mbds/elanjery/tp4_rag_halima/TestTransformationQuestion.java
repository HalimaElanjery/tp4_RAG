
package ht.fds.mbds.elanjery.tp4_rag_halima;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.data.embedding.Embedding;

import dev.langchain4j.memory.chat.MessageWindowChatMemory;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;

import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.transformer.CompressingQueryTransformer;

import dev.langchain4j.service.AiServices;

import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.List;
import java.util.Scanner;

public class TestTransformationQuestion {

    public static void main(String[] args) {

        // PHASE 1 : Chargement du PDF
        System.out.println("Chargement du document PDF...");

        ApacheTikaDocumentParser parser =
                new ApacheTikaDocumentParser();

        Document document =
                ClassPathDocumentLoader.loadDocument("rag.pdf", parser);

        System.out.println("Document chargé avec succès.");

        // PHASE 2 : Découpage du document
        DocumentSplitter splitter =
                DocumentSplitters.recursive(300, 30);

        List<TextSegment> segments = splitter.split(document);

        System.out.println("Nombre de segments : " + segments.size());

        // PHASE 3 : Création des embeddings
        System.out.println("\nCréation du modèle d'embedding...");

        EmbeddingModel embeddingModel =
                new AllMiniLmL6V2EmbeddingModel();

        List<Embedding> embeddings =
                embeddingModel.embedAll(segments).content();

        System.out.println("Nombre d'embeddings : " + embeddings.size());

        // PHASE 4 : Stockage vectoriel
        System.out.println("\nCréation de la base vectorielle...");

        InMemoryEmbeddingStore<TextSegment> embeddingStore =
                new InMemoryEmbeddingStore<>();

        embeddingStore.addAll(embeddings, segments);

        System.out.println("Stockage terminé.");

        // PHASE 5 : Création du ContentRetriever
        System.out.println("\nCréation du moteur de recherche...");

        ContentRetriever contentRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(embeddingStore)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        System.out.println("Moteur de recherche créé.");

        // PHASE 6 : Initialisation de Gemini avec logging
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

        // PHASE 7 : Création de la mémoire conversationnelle
        MessageWindowChatMemory chatMemory =
                MessageWindowChatMemory.withMaxMessages(10);

        // PHASE 8 : Création du QueryTransformer
        System.out.println("\nCréation du QueryTransformer...");

        CompressingQueryTransformer queryTransformer =
                CompressingQueryTransformer.builder()
                        .chatModel(chatModel)
                        .build();

        System.out.println("QueryTransformer créé.");

        // PHASE 9 : Création du RetrievalAugmentor
        RetrievalAugmentor retrievalAugmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryTransformer(queryTransformer)
                        .contentRetriever(contentRetriever)
                        .build();

        System.out.println("RetrievalAugmentor créé.");

        // PHASE 10 : Création de l'assistant
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .chatMemory(chatMemory)
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        System.out.println("\nAssistant avec Query Transformation prêt !");

        // PHASE 11 : Conversation
        conversationAvec(assistant);
    }

    private static void conversationAvec(Assistant assistant) {

        try (Scanner scanner = new Scanner(System.in)) {

            System.out.println("\n======================================");
            System.out.println("   ASSISTANT RAG AVEC QUERY TRANSFORMER");
            System.out.println("======================================");

            System.out.println("Pose tes questions sur le cours.");
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
