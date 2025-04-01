# GraphRAG PDF Processing

This repository contains a Python script for processing PDF documents (specifically SEC Form 10-K filings) using Graph Retrieval Augmented Generation (GraphRAG) techniques. The implementation uses Neo4j for graph storage and OpenAI for embeddings and language model capabilities.

## Repository Contents

- **graphrag_script.py**: Main script for processing PDFs and querying the knowledge graph
- **data/**: Directory containing sample data
  - **form10k-clean_short/**: Directory containing sample Form 10-K PDF files
  - **cik-10k-urls_short_list.csv**: Mapping of CUSIPs to CIKs and Form 10-K URLs
  - **Company_Financial_Statements.csv**: Sample financial statement data
  - **Asset_Manager_Holdings.csv**: Sample asset manager holdings data

## Features

- Direct PDF loading using `PdfLoader` from neo4j_graphrag
- Text chunking with `FixedSizeSplitter` for optimal context size
- Embedding generation with `TextChunkEmbedder`
- Metadata extraction (CIK, CUSIP) from PDF content
- Graph creation in Neo4j with proper relationships
- Support for various retrieval methods (vector, keyword, graph, hybrid)
- Loading of company financial data and asset manager holdings from CSV files

## Requirements

- Python 3.8+
- Neo4j AuraDB instance or local Neo4j server
- OpenAI API key
- Required Python packages (see below)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/alisoncossette/GraphRAG_Ebook.git
   cd GraphRAG_Ebook
   ```

2. Install required packages:
   ```
   pip install neo4j>=5.14.0 neo4j-graphrag>=0.2.0 langchain>=0.1.0 python-dotenv>=1.0.0 requests>=2.31.0 openai>=1.12.0 tqdm>=4.66.0 pypdf pdfplumber pandas
   ```

3. Create a `.env` file in the root directory with the following content:
   ```
   NEO4J_URI=neo4j+s://your-instance-id.databases.neo4j.io
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your-password
   OPENAI_API_KEY=your-openai-api-key
   PDF_PATH=data/form10k-clean_short
   ```

## Usage

### Command-Line Options

```
usage: graphrag_script.py [-h] [--pdf PDF] [--dir DIR] [--query QUERY]
                          [--retriever {vector,keyword,graph,hybrid}] [--load]
                          [--load-companies] [--load-holdings] [--use-lexical-graph]

GraphRAG PDF Processing and Querying Tool

options:
  -h, --help            show this help message and exit
  --pdf PDF             Path to a single PDF file to process
  --dir DIR             Directory containing PDF files to process
  --query QUERY         Query the knowledge graph
  --retriever {vector,keyword,graph,hybrid}
                        Retriever type to use for querying (default: hybrid)
  --load                Load all PDFs from the default directory
  --load-companies      Load company financials from CSV
  --load-holdings       Load asset manager holdings from CSV
  --use-lexical-graph   Use lexical graph builder instead of default
```

### Process a Single PDF

```
python graphrag_script.py --pdf "data/form10k-clean_short/0000320193-23-000106.pdf"
```

### Process All PDFs in a Directory

```
python graphrag_script.py --load
```

### Process All PDFs with Lexical Graph

```
python graphrag_script.py --load --use-lexical-graph
```

### Load Company Financial Data

```
python graphrag_script.py --load-companies
```

### Load Asset Manager Holdings Data

```
python graphrag_script.py --load-holdings
```

### Query the Knowledge Graph

```
python graphrag_script.py --query "What was Apple's revenue in 2023?" --retriever hybrid
```

## How It Works

1. **PDF Loading**: The script loads PDF documents using the `PdfLoader` component.
2. **Text Extraction**: Text is extracted from the PDFs using PyPDF2 or pdfplumber.
3. **Metadata Extraction**: CIK and CUSIP identifiers are extracted using regex patterns.
4. **Text Chunking**: The extracted text is split into manageable chunks using `FixedSizeSplitter`.
5. **Embedding Generation**: Text chunks are embedded using OpenAI embeddings.
6. **Graph Creation**: The script creates Document, CIK, CUSIP, and Chunk nodes in Neo4j.
7. **Relationship Establishment**: Relationships are created between nodes (e.g., Document-CIK, Document-CUSIP, Chunk-Document).
8. **CSV Data Loading**: Company financial data and asset manager holdings can be loaded from CSV files.
9. **Querying**: The knowledge graph can be queried using various retrieval methods.

## Neo4j Schema

- **Document**: Nodes representing PDF documents
- **CIK**: Nodes representing Central Index Key identifiers
- **CUSIP**: Nodes representing Committee on Uniform Securities Identification Procedures identifiers
- **Chunk**: Nodes containing text chunks with embeddings
- **Company**: Nodes representing companies with financial data
- **AssetManager**: Nodes representing asset management firms
- **HOLDS**: Relationships between asset managers and companies they invest in

## Sample .env File

```
# Neo4j Connection Details
NEO4J_URI=neo4j+s://your-instance-id.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-secure-password

# OpenAI API Key
OPENAI_API_KEY=sk-your-openai-api-key

# PDF Directory Path
PDF_PATH=data/form10k-clean_short
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
