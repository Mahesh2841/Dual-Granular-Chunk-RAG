# Dual-Granular-Chunk-RAG

  	  
Abstract— This project presents a novel framework that enhances large language models (LLMs) by integrating multi-granular text segmentation with ensemble retrieval techniques. By leveraging dual chunking strategies—a broad context with larger overlapping segments and a narrow context with more precise, smaller segments—we generate complementary embedding representations that are stored in separate FAISS indices. At query time, our system retrieves semantically relevant chunks from both indices and fuses them to create a dual-layered context for the LLM, thus bridging the trade-off between recall and precision. In this paper, we provide detailed mathematical formulations, algorithmic complexity analysis, and extensive experimental results. Our framework not only demonstrates improvements in retrieval metrics but also offers new insights into context augmentation for LLMs.

![iee paper ff](https://github.com/user-attachments/assets/e10013ed-bd33-482e-9715-99299c37db2e)

For a document T of length L, the ith chunk in a configuration with chunk size c and overlap o is defined as: 

Ti = T [i × (c − o): i × (c − o) + c], i ∈ {0, 1, …, N−1}, 

with the total number of chunks N given by: 

 N= [c – o / L − o]  
 
 This overlapping scheme ensures continuity between chunks, which is crucial for preserving context at boundaries [15].  

 Let femb : Text → Rd denote the embedding function. For each chunk Ti , we compute: 
 
ei = femb (Ti), 
    
where ei ∈ Rd and d is the embedding dimension. The embeddings are computed in batches to leverage parallel processing capabilities. 

The computed embeddings for the broad and narrow contexts are organized into matrices: 

  Eb = [e1b, e2b, …, eNbb], En=[e1n,e2n,…,eNnn], 
       
where Nb and Nn denote the number of chunks for the broad and narrow configurations, respectively. We then construct separate FAISS indices Ib and In using an L2 distance metric. Given a query embedding eq , the similarity score between eq and a chunk embedding ei  is:

d (eq, ei) = ∥eq − ei∥22 

At query time, a user-provided query Q is first converted to its embedding eq using the same embedding function femb. We then query both indices independently: 

Rb= FAISS-Query (Ib, eq, kb), Rn=FAISS-Query (In, eq, kn). 

Here, Rb  and Rn represent the sets of retrieved chunks from the broad and narrow indices, respectively. 

To merge these retrievals, we define a fusion function F that combines the outputs into a single context Cf for the LLM. A simple yet effective fusion strategy is the union of the two sets followed by deduplication: 

Cf = F (Rb, Rn) = ⋃ {r ∣ r ∈ Rb ∪ Rn}. 

More sophisticated fusion mechanisms (e.g., weighted averaging or attention-based merging) can also be considered, but our experiments demonstrate that even a straightforward union yields significant improvements 







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
