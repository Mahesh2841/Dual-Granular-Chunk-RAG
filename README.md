# Dual-Granular-Chunk-RAG

  	  
Abstract— This project presents a novel framework that enhances large language models (LLMs) by integrating multi-granular text segmentation with ensemble retrieval techniques. By leveraging dual chunking strategies—a broad context with larger overlapping segments and a narrow context with more precise, smaller segments—we generate complementary embedding representations that are stored in separate FAISS indices. At query time, our system retrieves semantically relevant chunks from both indices and fuses them to create a dual-layered context for the LLM, thus bridging the trade-off between recall and precision. In this paper, we provide detailed mathematical formulations, algorithmic complexity analysis, and extensive experimental results. Our framework not only demonstrates improvements in retrieval metrics but also offers new insights into context augmentation for LLMs.

![iee paper ff](https://github.com/user-attachments/assets/e10013ed-bd33-482e-9715-99299c37db2e)




Enhancing Retrieval-Augmented Generation via
Dual-Granularity Document Indexing, IEEE International Conference on Computing for Sustainability and Intelligent Future (CompSIF 2025), Bangalore Section
• Designed and implemented a novel dual-granularity retrieval-augmentation strategy integrating
multi-granular chunking and ensemble semantic indexing (FAISS-based vector search), significantly improving the precision (87%) and recall (73%) of retrieval-augmented Large Language
Models (LLMs).
• Proposed an ensemble fusion mechanism that merges broad and narrow contexts into a unified
retrieval pipeline, providing a comprehensive dual-layered context that enhances the accuracy
and contextual relevance of LLM-generated responses.
• Awarded Best Paper for technical innovation and significant contribution to knowledge retrieval
and NLP systems at CompSIF 2025, IEEE Bangalore Section.
