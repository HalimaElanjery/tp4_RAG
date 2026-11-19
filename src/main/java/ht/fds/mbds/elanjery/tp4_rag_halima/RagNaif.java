
package ht.fds.mbds.elanjery.tp4_rag_halima;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;

import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;

import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.List;

public class RagNaif {

    public static void main(String[] args) {

        // PHASE 1 : Chargement du document PDF

        System.out.println("Chargement du document PDF...");

        ApacheTikaDocumentParser parser =
                new ApacheTikaDocumentParser();

        Document document = ClassPathDocumentLoader.loadDocument(
                "rag.pdf",
                parser
        );

        System.out.println("Document chargé avec succès.");

        // Découpage du document

        DocumentSplitter splitter =
                DocumentSplitters.recursive(300, 30);

        List<TextSegment> segments = splitter.split(document);

        System.out.println("Nombre de segments générés : "
                + segments.size());

        afficherContenusSegments(segments, 0, 6);


        // PHASE 2 : Création des embeddings

        System.out.println("\nCréation du modèle d'embedding...");

        EmbeddingModel embeddingModel =
                new AllMiniLmL6V2EmbeddingModel();

        System.out.println("Modèle d'embedding créé.");


        // Transformation des segments en embeddings

        System.out.println("\nTransformation des segments en embeddings...");

        List<Embedding> embeddings =
                embeddingModel.embedAll(segments).content();

        System.out.println("Nombre d'embeddings générés : "
                + embeddings.size());


        // PHASE 3 : Stockage des embeddings

        System.out.println("\nCréation de la base vectorielle...");

        InMemoryEmbeddingStore<TextSegment> embeddingStore =
                new InMemoryEmbeddingStore<>();

        embeddingStore.addAll(embeddings, segments);

        System.out.println("Stockage terminé avec succès.");

        System.out.println("\n========== RÉSUMÉ ==========");
        System.out.println("Nombre de segments : " + segments.size());
        System.out.println("Nombre d'embeddings : " + embeddings.size());
        System.out.println("Base vectorielle initialisée.");
    }


    // Méthode d'affichage des segments

    private static void afficherContenusSegments(
            List<TextSegment> segments,
            int debut,
            int fin
    ) {

        System.out.println("\n========== AFFICHAGE DES SEGMENTS ==========");

        for (int i = debut; i < Math.min(fin, segments.size()); i++) {

            System.out.println("\n======================================");
            System.out.println("Segment " + i);

            System.out.println("\nContenu du segment :");
            System.out.println(segments.get(i).text());

            System.out.println("\nMétadonnées :");
            System.out.println(segments.get(i).metadata());
        }
    }
}
