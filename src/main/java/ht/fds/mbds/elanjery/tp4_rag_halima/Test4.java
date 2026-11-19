
package ht.fds.mbds.elanjery.tp4_rag_halima;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Test4 {

    public static void main(String[] args) {

        // 1. Chargement du document PDF
        Document document = ClassPathDocumentLoader.loadDocument(
                "rag.pdf",
                new ApacheTikaDocumentParser()
        );

        // 2. Découpage du document en segments
        List<TextSegment> segments = DocumentSplitters.recursive(300, 30)
                .split(document);

        // 3. Création du modèle d'embeddings
        EmbeddingModel embeddingModel =
                new AllMiniLmL6V2EmbeddingModel();

        // 4. Génération des embeddings
        List<Embedding> embeddings =
                embeddingModel.embedAll(segments).content();

        System.out.println("Nombre d'embeddings : " + embeddings.size());

        // 5. Stockage vectoriel
        EmbeddingStore<TextSegment> embeddingStore =
                new InMemoryEmbeddingStore<>();

        embeddingStore.addAll(embeddings, segments);

        System.out.println("Stockage vectoriel terminé.");

        // 6. Création du retriever PDF
        ContentRetriever pdfContentRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(embeddingStore)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        System.out.println("ContentRetriever PDF créé.");

        // 7. Création du moteur de recherche Tavily
        String tavilyApiKey = System.getenv("TAVILY_API_KEY");

        if (tavilyApiKey == null || tavilyApiKey.isBlank()) {
            throw new IllegalStateException(
                    "La variable d'environnement TAVILY_API_KEY n'est pas configurée."
            );
        }

        WebSearchEngine webSearchEngine =
                TavilyWebSearchEngine.builder()
                        .apiKey(tavilyApiKey)
                        .build();

        System.out.println("Moteur de recherche Tavily créé.");

        // 8. Création du retriever Web
        ContentRetriever webContentRetriever =
                WebSearchContentRetriever.builder()
                        .webSearchEngine(webSearchEngine)
                        .build();

        System.out.println("ContentRetriever Web créé.");

        // 9. Création du modèle Gemini avec logging
        String geminiApiKey = System.getenv("GEMINI_KEY");

        if (geminiApiKey == null || geminiApiKey.isBlank()) {
            throw new IllegalStateException(
                    "La variable d'environnement GEMINI_KEY n'est pas configurée."
            );
        }

        ChatModel chatModel = GoogleGenAiChatModel.builder()
                .apiKey(geminiApiKey)
                .modelName("gemini-3.5-flash-lite")
                .logRequests(true)
                .logResponses(true)
                .build();

        System.out.println("Modèle Gemini créé avec logging.");

        // 10. Ancienne version : DefaultQueryRouter
        /*
         * Ce routeur utilise systématiquement les deux ContentRetriever.
         * Même une question sans rapport avec le PDF déclenche donc
         * une recherche dans le PDF et une recherche sur le Web.
         *
         * QueryRouter queryRouter = new DefaultQueryRouter(
         *         pdfContentRetriever,
         *         webContentRetriever
         * );
         */

        // 11. Nouveau routeur intelligent personnalisé
        QueryRouter queryRouter = new QueryRouter() {

            @Override
            public List<ContentRetriever> route(Query query) {

                String question = query.text();

                String prompt = """
                        Tu es un routeur de recherche pour un système RAG.

                        Tu dois choisir la source la plus adaptée à la question.

                        SOURCE PDF :
                        Le PDF contient un support de cours portant notamment
                        sur le RAG, le fine-tuning, l'intelligence artificielle,
                        le machine learning et les concepts associés.

                        SOURCE WEB :
                        Le Web sert à rechercher les actualités, les événements
                        récents, les résultats électoraux, les informations
                        actuelles et les informations extérieures au support.

                        RÈGLES DE DÉCISION :

                        1. Si la question porte sur une notion expliquée dans
                        le support de cours, choisis PDF uniquement.

                        2. Si la question porte sur une actualité, un événement
                        récent ou une information extérieure au cours,
                        choisis WEB uniquement.

                        3. Choisis BOTH uniquement si la question nécessite
                        réellement de consulter à la fois le cours et le Web.

                        Réponds obligatoirement avec un seul de ces mots :
                        PDF
                        WEB
                        BOTH

                        Ne donne aucune explication.

                        Question : %s
                        """.formatted(question);

                // chatModel.chat(prompt) renvoie directement une String.
                String decision = chatModel.chat(prompt)
                        .trim()
                        .toUpperCase();

                System.out.println(
                        "\nQuestion analysée par le routeur : " + question
                );

                System.out.println("Décision du routeur : " + decision);

                List<ContentRetriever> selectedRetrievers =
                        new ArrayList<>();

                if (decision.contains("BOTH")) {

                    selectedRetrievers.add(pdfContentRetriever);
                    selectedRetrievers.add(webContentRetriever);

                } else if (decision.contains("PDF")) {

                    selectedRetrievers.add(pdfContentRetriever);

                } else if (decision.contains("WEB")) {

                    selectedRetrievers.add(webContentRetriever);

                } else {

                    selectedRetrievers.add(webContentRetriever);

                    System.out.println(
                            "Décision non reconnue : sélection du Web par défaut."
                    );
                }

                return selectedRetrievers;
            }
        };

        System.out.println("Routeur intelligent créé.");

        // 12. Création du RetrievalAugmentor
        RetrievalAugmentor retrievalAugmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryRouter(queryRouter)
                        .build();

        System.out.println("RetrievalAugmentor créé.");

        // 13. Création de l'assistant
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        System.out.println("Assistant créé.");

        // 14. Conversation avec l'assistant
        Scanner scanner = new Scanner(System.in);

        System.out.println(
                "\nPose tes questions (tape 'exit' pour quitter)."
        );

        while (true) {

            System.out.print("\nPose ta question : ");

            String question = scanner.nextLine();

            if (question.equalsIgnoreCase("exit")) {
                break;
            }

            String response = assistant.chat(question);

            System.out.println("\nRéponse : " + response);
        }

        scanner.close();
    }
}
