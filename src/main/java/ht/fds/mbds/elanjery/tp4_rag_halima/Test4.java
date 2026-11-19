
package ht.fds.mbds.elanjery.tp4_rag_halima;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;

import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;

import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;

import java.util.List;
import java.util.Scanner;

public class Test4 {

    public static void main(String[] args) {

        // 1. Charger le document PDF
        Document document = ClassPathDocumentLoader.loadDocument(
                "rag.pdf",
                new ApacheTikaDocumentParser()
        );

        System.out.println("Document chargé : rag.pdf");

        // 2. Découper le document en segments
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);

        List<TextSegment> segments = splitter.split(document);

        System.out.println("Nombre de segments : " + segments.size());

        // 3. Créer le modèle d'embedding
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // 4. Générer les embeddings
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        System.out.println("Nombre d'embeddings : " + embeddings.size());

        // 5. Créer le stockage vectoriel
        EmbeddingStore<TextSegment> embeddingStore =
                new InMemoryEmbeddingStore<>();

        embeddingStore.addAll(embeddings, segments);

        System.out.println("Stockage vectoriel terminé.");

        // 6. Créer le ContentRetriever du PDF
        ContentRetriever pdfContentRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(embeddingStore)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        System.out.println("ContentRetriever PDF créé.");

        // 7. Créer le moteur de recherche Web Tavily
        String tavilyApiKey = System.getenv("TAVILY_API_KEY");

        if (tavilyApiKey == null || tavilyApiKey.isBlank()) {
            throw new IllegalStateException(
                    "La variable d'environnement TAVILY_API_KEY n'est pas configurée."
            );
        }

        WebSearchEngine webSearchEngine = TavilyWebSearchEngine.builder()
                .apiKey(tavilyApiKey)
                .build();

        System.out.println("Moteur de recherche Tavily créé.");

        // 8. Créer le ContentRetriever Web
        ContentRetriever webContentRetriever =
                WebSearchContentRetriever.builder()
                        .webSearchEngine(webSearchEngine)
                        .build();

        System.out.println("ContentRetriever Web créé.");

        // 9. Créer le QueryRouter avec les deux retrievers
        QueryRouter queryRouter = new DefaultQueryRouter(
                pdfContentRetriever,
                webContentRetriever
        );

        System.out.println("DefaultQueryRouter créé.");

        // 10. Créer le RetrievalAugmentor
        RetrievalAugmentor retrievalAugmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryRouter(queryRouter)
                        .build();

        System.out.println("RetrievalAugmentor créé.");

        // 11. Créer le modèle Gemini avec logging
        String geminiApiKey = System.getenv("GEMINI_KEY");

        if (geminiApiKey == null || geminiApiKey.isBlank()) {
            throw new IllegalStateException(
                    "La variable d'environnement GEMINI_KEY n'est pas configurée."
            );
        }

        GoogleGenAiChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(geminiApiKey)
                .modelName("gemini-3.5-flash-lite")
                .logRequests(true)
                .logResponses(true)
                .build();

        System.out.println("Modèle Gemini créé avec logging.");

        // 12. Créer l'assistant avec le RetrievalAugmentor
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        System.out.println("Assistant créé.");

        // 13. Boucle de conversation
        Scanner scanner = new Scanner(System.in);

        System.out.println("\nPose tes questions (tape 'exit' pour quitter).");

        while (true) {

            System.out.print("\nPose ta question : ");
            String question = scanner.nextLine();

            if (question.equalsIgnoreCase("exit")) {
                System.out.println("Fin de la conversation.");
                break;
            }

            String reponse = assistant.chat(question);

            System.out.println("\nRéponse : " + reponse);
        }

        scanner.close();
    }
}
