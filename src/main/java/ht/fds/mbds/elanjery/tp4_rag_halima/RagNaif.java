
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

import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;

import dev.langchain4j.service.AiServices;

import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.List;
import java.util.Scanner;

public class RagNaif {

    public static void main(String[] args) {

        // PHASE 1 : Chargement et découpage du PDF

        System.out.println("Chargement du document PDF...");

        ApacheTikaDocumentParser parser =
                new ApacheTikaDocumentParser();

        Document document = ClassPathDocumentLoader.loadDocument(
                "rag.pdf",
                parser
        );

        System.out.println("Document chargé avec succès.");

        DocumentSplitter splitter =
                DocumentSplitters.recursive(300, 30);

        List<TextSegment> segments = splitter.split(document);

        System.out.println("Nombre de segments : " + segments.size());


        // PHASE 2 : Création des embeddings

        System.out.println("\nCréation du modèle d'embedding...");

        EmbeddingModel embeddingModel =
                new AllMiniLmL6V2EmbeddingModel();

        List<Embedding> embeddings =
                embeddingModel.embedAll(segments).content();

        System.out.println("Nombre d'embeddings : " + embeddings.size());


        // PHASE 3 : Stockage vectoriel

        System.out.println("\nCréation de la base vectorielle...");

        InMemoryEmbeddingStore<TextSegment> embeddingStore =
                new InMemoryEmbeddingStore<>();

        embeddingStore.addAll(embeddings, segments);

        System.out.println("Stockage terminé.");


        // PHASE 4 : Création du moteur de recherche RAG

        System.out.println("\nCréation du moteur de recherche...");

        ContentRetriever contentRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(embeddingStore)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        System.out.println("Moteur de recherche créé.");


        // PHASE 5 : Création du modèle Gemini

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
                .build();

        System.out.println("Gemini initialisé.");


        // PHASE 6 : Création de la mémoire conversationnelle

        MessageWindowChatMemory chatMemory =
                MessageWindowChatMemory.withMaxMessages(10);


        // PHASE 7 : Création de l'assistant RAG

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .chatMemory(chatMemory)
                .contentRetriever(contentRetriever)
                .build();

        System.out.println("\nAssistant RAG prêt !");


        // PHASE 8 : Interaction avec l'utilisateur

        Scanner scanner = new Scanner(System.in);

        System.out.println("\n==================================");
        System.out.println("       ASSISTANT RAG");
        System.out.println("==================================");
        System.out.println("Pose tes questions sur le cours.");
        System.out.println("Tape 'exit' pour quitter.");

        while (true) {

            System.out.print("\nVotre question : ");

            String question = scanner.nextLine();

            if (question.equalsIgnoreCase("exit")) {
                System.out.println("Fin de la conversation.");
                break;
            }

            if (question.isBlank()) {
                continue;
            }

            String reponse = assistant.chat(question);

            System.out.println("\nRéponse de Gemini :");
            System.out.println(reponse);
        }

        scanner.close();
    }
}
