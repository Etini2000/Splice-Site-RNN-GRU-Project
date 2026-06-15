# Saccharomyces cerevisiae Chromosome 4 Splice Site RNN (GRU) AI Model Project

This project is an end-to-end deep learning pipeline built from scratch in PyTorch to classify eukaryotic genomic data. This project programmatically downloads Chromosome IV of Baker's Yeast (*Saccharomyces cerevisiae*), processes millions of raw nucleotides, and uses a **Gated Recurrent Unit (GRU)** network to discover true biological splice-site motifs hidden in genetic noise.

---

## Overview
So in a DNA sequence, there are Introns ('useless' codes...), and Exons (useful codes...). The boundary where an Exon ends and an Intron begins is called a SPLICE SITE. In almost all complex organisms, the exact boundary where an Intron begins almost always starts with two specific nucleotide letters: GT (called the donor splice site). In this project, we will use Chromosome 4 of Saccharomyces cerevisiae.

The aim of this project is to build and evaluate an AI model that can identify a splice site.
Remember GT appear randomly millions of times across a genome just by pure chance. So here, we will differntiate a true splice site sequence from a random DNA sequence that happens to contain a GT but are not real splice sites.

AI models cannot read raw text; they require clean, balanced numerical inputs. This repository demonstrates a complete machine learning lifecycle for sequence-based biological data:

1. **Automated Data Ingestion:** Connects to the NCBI database to stream real genomic data.
2. **Feature Engineering:** Slides a context window across the genome to extract real motifs and balanced background noise sites.
3. **Numerical Vectorization:** Converts genetic text (`A`, `C`, `G`, `T`) into binary 3D matrices using One-Hot Encoding.
4. **Recurrent Modeling:** Uses a PyTorch GRU architecture to read sequences sequentially and maintain an internal memory state.
5. **Optimization:** Trains and evaluates the model over 100 epochs, tracking loss minimization.

---

## Pipeline Architecture & Code Walkthrough

### 1. Data Retrieval (`download_yeast_genome`)
Uses the **Biopython Entrez** library to fetch RefSeq Accession `NC_001136` (Yeast Chromosome IV). Includes smart local caching to prevent redundant internet downloads.

### 2. Dataset Balancing (`extract_splice_windows`)
Scans the ~1.5 million DNA letters. True splice donor sites (`GTATGT` / `GTAGGT`) are labeled as **Class 1**. General background `GT` sites are filtered and isolated as **Class 0**. To prevent model bias, the majority class is randomly downsampled to create a perfectly balanced dataset.

### 3. PyTorch Conversion (`one_hot_encode_dna`)
Maps character arrays to numeric vectors using a custom layout dictionary mapping nucleotides to active dimensions. The resulting array is stacked into a 3D PyTorch Tensor shaped as `(Samples, Window Length, One-Hot Dimensions)`.

### 4. Recurrent Neural Network (`GeneFinderGRU`)
Implements a custom sub-classed `nn.Module` containing:
- **GRU Layer:** Processes 4 input features across a 40-step sequence length, building a 32-dimensional internal memory state.
- **Linear Layer:** Condenses the final memory vector into a single logit prediction score.

---

## Performance & Optimization

The network was trained for 100 epochs using **Binary Cross Entropy with Logits Loss (`BCEWithLogitsLoss`)** and the **Adam Optimizer** (Learning Rate = 0.001). 

- **Baseline Challenge:** Because the dataset is perfectly balanced, random coin-flip guessing yields exactly 50% accuracy.
- **Result:** The model successfully drops optimization loss and breaks past the random guessing barrier, proving the GRU effectively learned sequence rules from raw genetic structures.


---

## Tech Stack & Dependencies

- **Language:** Python 3
- **Deep Learning Framework:** PyTorch
- **Scientific Computing:** NumPy
- **Bioinformatics Toolkit:** Biopython (`Bio.SeqIO`)
- **Data Visualization:** Matplotlib

---
