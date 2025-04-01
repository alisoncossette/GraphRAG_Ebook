#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GraphRAG Script - PDF to Knowledge Graph
This script processes PDF documents and creates a knowledge graph in Neo4j using GraphRAG.
Supports the GraphRAG ebook article and incorporates comments from GraphRAG regroup.
"""

import os
import sys
import re
import json
import pandas as pd
from pathlib import Path
import PyPDF2
from neo4j import GraphDatabase
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import neo4j_graphrag
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from pdfplumber import open as pdf_open
import dotenv
from tqdm import tqdm
import requests
import csv
import io
import pdfplumber
import traceback
import asyncio
import time

# Load environment variables from .env file if it exists
dotenv.load_dotenv()

# Neo4j connection parameters - using the correct AuraDB instance from memory
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")  # Updated to match .env variable name
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")  # Should be set in .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Should be set in .env file

# Note: the OPENAI_API_KEY must be in the env vars
llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})
pdf_dir = os.getenv("PDF_PATH")

def ensure_vector_index_exists(driver, index_name="chunkEmbeddings", dimension=1536):
    """
    Ensure that a vector index exists in Neo4j.
    
    Args:
        driver: Neo4j driver instance
        index_name: Name of the vector index
        dimension: Dimension of the embeddings
    """
    try:
        # Check if the index exists
        result = driver.execute_query(
            """
            SHOW INDEXES
            YIELD name, type
            WHERE name = $index_name AND type = 'VECTOR'
            RETURN count(*) > 0 AS exists
            """,
            index_name=index_name
        )
        
        index_exists = result[0][0]["exists"]
        
        if not index_exists:
            print(f"Vector index '{index_name}' not found. Creating it...")
            
            # Create the vector index
            driver.execute_query(
                f"""
                CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                FOR (c:Chunk) ON (c.embeddings)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {dimension},
                    `vector.similarity_function`: 'cosine'
                }}}}
                """
            )
            print(f"Vector index '{index_name}' created successfully")
        else:
            print(f"Vector index '{index_name}' already exists")
        
        return True
    except Exception as e:
        print(f"Error ensuring vector index exists: {str(e)}")
        return False

def setup_neo4j_connection():
    """Set up connection to Neo4j database."""
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Create Neo4j driver instance
            uri = os.getenv("NEO4J_URI")
            user = os.getenv("NEO4J_USERNAME")
            password = os.getenv("NEO4J_PASSWORD")
            
            # Create driver with connection timeout and retry settings
            driver = GraphDatabase.driver(
                uri, 
                auth=(user, password),
                max_connection_lifetime=300,  # 5 minutes
                max_transaction_retry_time=15.0,  # 15 seconds
                connection_acquisition_timeout=60.0  # 60 seconds
            )
            
            # Test connection using driver.execute_query (Neo4j 5.x API)
            result = driver.execute_query("RETURN 1 AS test")
            print(f"Connection to Neo4j successful: {result[0][0]['test']}")
                
            # Check if the vector index exists and create it if needed
            ensure_vector_index_exists(driver)
            
            return driver
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error connecting to Neo4j (attempt {attempt+1}/{max_retries}): {str(e)}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Failed to connect to Neo4j after {max_retries} attempts: {str(e)}")
                return None

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path, timeout=300):
    """
    Extract text from a PDF file using multiple methods with timeout protection.
    
    Args:
        pdf_path (str): Path to the PDF file
        timeout (int): Timeout in seconds for extraction
    
    Returns:
        str: Extracted text from the PDF
    """
    print(f"Extracting text from {pdf_path}")
    
    # First try PyPDF2
    try:
        print(f"Trying PyPDF2 for {pdf_path}")
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            print(f"Processing {num_pages} pages from {pdf_path} with PyPDF2")
            
            # Process pages with timeout protection
            text = ""
            for i in range(num_pages):
                try:
                    page = reader.pages[i]
                    text += page.extract_text() + "\n"
                except Exception as e:
                    print(f"Error extracting text from page {i} with PyPDF2: {e}")
            
            # Check if we got any text
            if text.strip():
                print(f"Successfully extracted {len(text)} characters from {pdf_path}")
                
                # Manually append metadata if it's in the filename
                file_name = os.path.basename(pdf_path)
                if file_name.startswith("000"):
                    # Try to extract CIK from the filename (SEC format)
                    match = re.search(r'(\d{10})-(\d{2})-(\d{6})', file_name)
                    if match:
                        accession_number = match.group(0)
                        # Add metadata footer if not already present
                        if "Metadata" not in text[-1000:]:
                            # Get the CIK and CUSIP from the CSV file if available
                            cik = None
                            cusip6 = None
                            source_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number.replace('-', '')}/{accession_number}"
                            
                            # Add metadata footer
                            metadata_footer = f"\n\nMetadata\nCIK: {cik if cik else 'Unknown'}\nCUSIP6: {cusip6 if cusip6 else 'Unknown'}\nSource: {source_url}\n"
                            print("CUSIP6: ", cusip6)
                            text += metadata_footer
                            print(f"Added metadata footer to the extracted text")
                
                return text
    except Exception as e:
        print(f"PyPDF2 extraction failed for {pdf_path}: {e}")
    
    # If PyPDF2 fails, try pdfplumber
    try:
        print(f"Trying pdfplumber for {pdf_path}")
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            print(f"Processing {num_pages} pages from {pdf_path} with pdfplumber")
            
            # Process pages with timeout protection
            for i in range(num_pages):
                try:
                    page = pdf.pages[i]
                    text += page.extract_text() + "\n"
                except Exception as e:
                    print(f"Error extracting text from page {i} with pdfplumber: {e}")
            
            # Check if we got any text
            if text.strip():
                print(f"Successfully extracted {len(text)} characters from {pdf_path}")
                
                # Manually append metadata if it's in the filename
                file_name = os.path.basename(pdf_path)
                if file_name.startswith("000"):
                    # Try to extract CIK from the filename (SEC format)
                    match = re.search(r'(\d{10})-(\d{2})-(\d{6})', file_name)
                    if match:
                        accession_number = match.group(0)
                        # Add metadata footer if not already present
                        if "Metadata" not in text[-1000:]:
                            # Get the CIK and CUSIP from the CSV file if available
                            cik = None
                            cusip6 = None
                            source_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number.replace('-', '')}/{accession_number}"
                            
                            # Add metadata footer
                            metadata_footer = f"\n\nMetadata\nCIK: {cik if cik else 'Unknown'}\nCUSIP6: {cusip6 if cusip6 else 'Unknown'}\nSource: {source_url}\n"
                            text += metadata_footer
                            print(f"Added metadata footer to the extracted text")
                
                return text
    except Exception as e:
        print(f"pdfplumber extraction failed for {pdf_path}: {e}")
    
    # If all methods fail, return empty string
    print(f"All PDF extraction methods failed for {pdf_path}")
    return ""

def extract_metadata(text):
    """
    Extract metadata like CIK and CUSIP from text, focusing on the end of the document.
    
    Returns:
        dict: Dictionary containing extracted metadata
    """
    # Get the last portion of the text where metadata is likely to be
    # Use the last 20% of the document or last 10,000 characters, whichever is smaller
    text_length = len(text)
    end_portion_size = min(int(text_length * 0.2), 10000)
    end_portion = text[-end_portion_size:] if text_length > end_portion_size else text
    
    # Print the last 1000 characters to check for metadata
    print("\n--- Last 1000 characters of the PDF text ---")
    print(text[-1000:])
    print("--- End of PDF text sample ---\n")
    
    metadata = {}
    
    # For performance, only search the beginning and end portions of the document
    # where metadata is typically found
    text_length = len(text)
    start_portion = text[:min(10000, text_length)]
    end_portion = text[max(0, text_length - 10000):]
    
    # Extract CIK
    cik_patterns = [
        r'CIK:?\s*(\d{10})',
        r'CIK:?\s*(\d{6})',  # Some documents use 6-digit CIK
        r'CIK\s*(?:Number|No\.?)?:?\s*(\d{10})',
        r'(?:^|\s)CIK[:\s]*(\d{10})(?:\s|$)',
        r'(?:^|\s)CIK[:\s]*(\d{6})(?:\s|$)',  # Some documents use 6-digit CIK
        r'Central\s+Index\s+Key:?\s*(\d{10})'
    ]
    
    for pattern in cik_patterns:
        matches = re.finditer(pattern, end_portion, re.IGNORECASE)
        for match in matches:
            cik = match.group(1).strip()
            if cik:
                metadata['cik'] = cik
                print(f"Found CIK: {cik}")
                break
        if 'cik' in metadata:
            break
    
    # Extract CUSIP6
    cusip6_patterns = [
        r'CUSIP6:?\s*([A-Z0-9]{6})',
        r'CUSIP\s*(?:Number|No\.?)?:?\s*([A-Z0-9]{6})[^A-Z0-9]',
        r'(?:^|\s)CUSIP[:\s]*([A-Z0-9]{6})[^A-Z0-9]',
        r'CUSIP[^A-Z0-9]*([A-Z0-9]{6})',  # More relaxed pattern
        r'CUSIP[^:]*:?\s*([A-Z0-9]{5,6})',  # Even more relaxed pattern
        r'CUSIP6:\s*([A-Z0-9]{5,6})'  # Format seen in test output
    ]
    
    # First search in the metadata section if it exists
    metadata_section = None
    metadata_patterns = [
        r'Metadata\s*(.*?)(?:\n\s*\n|\Z)',
        r'CIK:.*?CUSIP.*?Source:'
    ]
    
    for pattern in metadata_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            metadata_section = match.group(0)
            print(f"Found metadata section: {metadata_section}")
            break
    
    # If we found a metadata section, search there first
    if metadata_section:
        for pattern in cusip6_patterns:
            matches = re.finditer(pattern, metadata_section, re.IGNORECASE)
            for match in matches:
                cusip6 = match.group(1).strip()
                if cusip6:
                    metadata['cusip6'] = cusip6
                    print(f"Found CUSIP6: {cusip6} in metadata section")
                    # Also print context around the match for debugging
                    start = max(0, match.start() - 30)
                    end = min(len(metadata_section), match.end() + 30)
                    context = metadata_section[start:end]
                    print(f"CUSIP6 context: ...{context}...")
                    break
            if 'cusip6' in metadata:
                break
    
    # If not found in metadata section, search the entire text
    if 'cusip6' not in metadata:
        for pattern in cusip6_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                cusip6 = match.group(1).strip()
                if cusip6:
                    metadata['cusip6'] = cusip6
                    print(f"Found CUSIP6: {cusip6} in full text")
                    # Also print context around the match for debugging
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    context = text[start:end]
                    print(f"CUSIP6 context: ...{context}...")
                    break
            if 'cusip6' in metadata:
                break
    
    # Extract full CUSIP (9 characters)
    cusip_patterns = [
        r'CUSIP:?\s*([A-Z0-9]{9})',
        r'CUSIP\s*(?:Number|No\.?)?:?\s*([A-Z0-9]{9})',
        r'(?:^|\s)CUSIP[:\s]*([A-Z0-9]{9})(?:\s|$)',
        r'CUSIP\s*(?:Number|No\.?)?:?\s*([A-Z0-9]{6})[^A-Z0-9]*([A-Z0-9]{3})'  # CUSIP with possible spacing
    ]
    
    for pattern in cusip_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            if len(match.groups()) == 1:
                cusip = match.group(1).strip()
                if cusip and len(cusip) == 9:
                    metadata['cusip'] = cusip
                    print(f"Found CUSIP: {cusip}")
                    break
            elif len(match.groups()) == 2:
                # Handle case where CUSIP is split into two groups
                cusip = match.group(1).strip() + match.group(2).strip()
                if cusip and len(cusip) == 9:
                    metadata['cusip'] = cusip
                    print(f"Found CUSIP: {cusip}")
                    break
        if 'cusip' in metadata:
            break
    
    # If we found CUSIP but not CUSIP6, extract it from CUSIP
    if 'cusip' in metadata and 'cusip6' not in metadata:
        metadata['cusip6'] = metadata['cusip'][:6]
        print(f"Derived CUSIP6: {metadata['cusip6']} from CUSIP: {metadata['cusip']}")
    
    return metadata

async def run_pipeline_async(pipeline, file_path):
    """Run the SimpleKGPipeline asynchronously."""
    # Pass the file_path parameter for PDF processing
    return await pipeline.run_async(file_path=file_path)

def execute_neo4j_query_with_retry(driver, query, **params):
    """Execute a Neo4j query with retry logic for connection issues."""
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            return driver.execute_query(query, **params)
        except Exception as e:
            if "defunct connection" in str(e) and attempt < max_retries - 1:
                print(f"Neo4j connection error (attempt {attempt+1}/{max_retries}): {str(e)}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                raise  # Re-raise the exception if it's not a connection issue or we've exhausted retries

def process_pdf(pdf_path, driver, llm, embedder, use_lexical_graph=False):
    """
    Process a PDF file and load it into Neo4j.
    
    Args:
        pdf_path: Path to the PDF file
        driver: Neo4j driver
        llm: Language model
        embedder: Embedder for text chunks
        use_lexical_graph: Not used anymore, kept for backward compatibility
    """
    try:
        # Extract text from PDF for metadata extraction only
        print(f"Extracting text from {pdf_path}")
        text = extract_text_from_pdf(pdf_path)
        
        if not text:
            print(f"Error: Could not extract text from {pdf_path}")
            return False
            
        print(f"Successfully extracted {len(text)} characters from {pdf_path}")
        
        # Extract metadata from text
        metadata = extract_metadata(text)
        print(f"Extracted metadata: {metadata}")
        
        # Create document node with metadata
        print(f"Creating Document node for {pdf_path}")
        driver.execute_query("""
            MERGE (d:Document {path: $path})
            SET d.type = 'pdf'
            RETURN d
        """, path=pdf_path)
        
        # Create CIK node if available and link to document
        if 'cik' in metadata:
            print(f"Creating CIK node with value: {metadata['cik']}")
            driver.execute_query("""
                MATCH (d:Document {path: $path})
                MERGE (c:CIK {value: $cik})
                MERGE (d)-[:HAS_CIK]->(c)
                RETURN d, c
            """, path=pdf_path, cik=metadata['cik'])
        
        # Create CUSIP node if available and link to document
        if 'cusip' in metadata:
            print(f"Creating CUSIP node with value: {metadata['cusip']}")
            try:
                result = driver.execute_query("""
                    MATCH (d:Document {path: $path})
                    MERGE (c:CUSIP {value: $cusip})
                    MERGE (d)-[:HAS_CUSIP]->(c)
                    RETURN d, c
                """, path=pdf_path, cusip=metadata['cusip'])
                print(f"CUSIP node creation result: {result.summary.counters}")
            except Exception as e:
                print(f"Error creating CUSIP node: {str(e)}")
                traceback.print_exc()
            
        # Create CUSIP node from CUSIP6 if available and link to document
        elif 'cusip6' in metadata:
            print(f"Creating CUSIP node with CUSIP6 value: {metadata['cusip6']}")
            try:
                result = driver.execute_query("""
                    MATCH (d:Document {path: $path})
                    MERGE (c:CUSIP {value: $cusip6})
                    MERGE (d)-[:HAS_CUSIP]->(c)
                    RETURN d, c
                """, path=pdf_path, cusip6=metadata['cusip6'])
                print(f"CUSIP node creation result: {result.summary.counters}")
                
                # Verify the CUSIP node was created
                verify_result = driver.execute_query("""
                    MATCH (c:CUSIP {value: $cusip6})
                    RETURN c
                """, cusip6=metadata['cusip6'])
                
                if verify_result.records:
                    print(f"Verified CUSIP node exists with value: {metadata['cusip6']}")
                else:
                    print(f"WARNING: CUSIP node with value {metadata['cusip6']} was not found after creation")
            except Exception as e:
                print(f"Error creating CUSIP node: {str(e)}")
                traceback.print_exc()
        
        # Process with direct components instead of SimpleKGPipeline
        print("Processing document with direct PDF loader, splitter, and embedder...")
        
        # Import necessary components
        from neo4j_graphrag.experimental.components.pdf_loader import PdfLoader
        from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
        from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder
        from neo4j_graphrag.experimental.components.types import DocumentInfo, TextChunk, TextChunks
        
        try:
            # 1. Load PDF
            print(f"Loading PDF with PdfLoader: {pdf_path}")
            pdf_loader = PdfLoader()
            try:
                pdf_document = asyncio.run(pdf_loader.run(pdf_path))
                print(f"PDF loaded successfully, text length: {len(pdf_document.text)}")
            except Exception as e:
                print(f"Error loading PDF with PdfLoader: {str(e)}")
                traceback.print_exc()
                # Use the text we already extracted as a fallback
                print("Using previously extracted text as fallback")
                from neo4j_graphrag.experimental.components.types import DocumentInfo, PdfDocument
                pdf_document = PdfDocument(
                    text=text,
                    document_info=DocumentInfo(path=pdf_path)
                )
            
            # 2. Split text into chunks
            print("Splitting text into chunks")
            text_splitter = FixedSizeSplitter(chunk_size=1000, chunk_overlap=200)
            try:
                text_chunks = asyncio.run(text_splitter.run(pdf_document.text))
                print(f"Text split into {len(text_chunks.chunks)} chunks")
            except Exception as e:
                print(f"Error splitting text: {str(e)}")
                traceback.print_exc()
                raise
            
            # 3. Embed chunks
            print("Embedding chunks")
            chunk_embedder = TextChunkEmbedder(embedder)
            try:
                embedded_chunks = asyncio.run(chunk_embedder.run(text_chunks))
                print(f"Successfully embedded {len(embedded_chunks.chunks)} chunks")
            except Exception as e:
                print(f"Error embedding chunks: {str(e)}")
                traceback.print_exc()
                raise
            
            # 4. Store chunks in Neo4j
            print(f"Storing {len(embedded_chunks.chunks)} chunks in Neo4j")
            
            # Create a batch of chunks to insert
            for i, chunk in enumerate(embedded_chunks.chunks):
                try:
                    # Create chunk node
                    driver.execute_query("""
                        MATCH (d:Document {path: $path})
                        CREATE (c:Chunk {
                            id: $chunk_id,
                            index: $index,
                            content: $content
                        })
                        SET c.embedding = $embedding
                        CREATE (c)-[:FROM_DOCUMENT]->(d)
                        RETURN c
                    """, 
                        path=pdf_path,
                        chunk_id=chunk.uid,
                        index=chunk.index,
                        content=chunk.text,
                        embedding=chunk.metadata["embedding"]
                    )
                    
                    # Create NEXT relationship between chunks
                    if i > 0:
                        prev_chunk_id = embedded_chunks.chunks[i-1].uid
                        driver.execute_query("""
                            MATCH (prev:Chunk {id: $prev_id})
                            MATCH (curr:Chunk {id: $curr_id})
                            CREATE (prev)-[:NEXT]->(curr)
                        """,
                            prev_id=prev_chunk_id,
                            curr_id=chunk.uid
                        )
                    
                    if i % 10 == 0 or i == len(embedded_chunks.chunks) - 1:
                        print(f"Stored {i+1}/{len(embedded_chunks.chunks)} chunks")
                        
                except Exception as e:
                    print(f"Error storing chunk {i}: {str(e)}")
                    traceback.print_exc()
                    # Continue with the next chunk
            
            print(f"Successfully processed {pdf_path} with direct components")
            return True
            
        except Exception as e:
            print(f"Error in direct component processing: {str(e)}")
            traceback.print_exc()
            print("Document metadata was still saved to Neo4j")
            return False
            
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {str(e)}")
        traceback.print_exc()
        return False

def process_multiple_pdfs(pdf_dir, driver, llm, embedder, use_lexical_graph=False):
    """Process all PDF files in a directory."""
    pdf_dir = Path(pdf_dir)
    
    if not pdf_dir.exists() or not pdf_dir.is_dir():
        print(f"Error: Directory {pdf_dir} does not exist or is not a directory")
        return
    
    # Get all PDF files in the directory
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
    
    # Process each PDF file
    for pdf_file in tqdm(pdf_files, desc="Processing PDF files"):
        print(f"\nProcessing {pdf_file}")
        process_pdf(str(pdf_file), driver, llm, embedder, use_lexical_graph)

def load_company_financials(csv_path, driver):
    """Load company financials from CSV file into Neo4j."""
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return False
    
    try:
        print(f"Loading company financials from {csv_path}")
        
        # Read the CSV file
        companies = []
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                companies.append(row)
        
        print(f"Found {len(companies)} companies in the CSV file")
        
        # Create company nodes and relationships in Neo4j
        with driver.session(database="neo4j") as session:
            # Create constraints if they don't exist
            try:
                session.run("""
                    CREATE CONSTRAINT company_cik_constraint IF NOT EXISTS
                    FOR (c:Company) REQUIRE c.cik IS UNIQUE
                """)
            except Exception as e:
                print(f"Warning: Could not create constraint: {e}")
            
            # Process each company
            for company in companies:
                try:
                    # Extract company data
                    cik = company.get('cik') or company.get('CIK')
                    name = company.get('name')
                    if isinstance(name, str) and (name.startswith('{') and name.endswith('}')):
                        # Parse the string as a Python literal
                        import ast
                        name = list(ast.literal_eval(name))[0]  # Take the first name
                    
                    cusip = company.get('cusip')
                    if isinstance(cusip, str) and (cusip.startswith('{') and cusip.endswith('}')):
                        # Parse the string as a Python literal
                        import ast
                        cusip = list(ast.literal_eval(cusip))[0]  # Take the first CUSIP
                    
                    form10k_url = company.get('form10KUrls')
                    
                    # Create company node
                    session.run("""
                        MERGE (c:Company {cik: $cik})
                        ON CREATE SET c.name = $name, c.cusip = $cusip, c.form10k_url = $form10k_url
                        ON MATCH SET c.name = $name, c.cusip = $cusip, c.form10k_url = $form10k_url
                        RETURN c
                    """, cik=cik, name=name, cusip=cusip, form10k_url=form10k_url)
                    
                    # Link company to its 10-K document if it exists
                    if form10k_url:
                        doc_title = os.path.basename(form10k_url)
                        session.run("""
                            MATCH (c:Company {cik: $cik})
                            MATCH (d:Document) 
                            WHERE d.title CONTAINS $doc_id
                            MERGE (c)-[:HAS_FILING]->(d)
                        """, cik=cik, doc_id=cik)
                
                except Exception as e:
                    print(f"Error processing company {company.get('cik')}: {e}")
        
        print(f"Successfully loaded company financials from {csv_path}")
        return True
        
    except Exception as e:
        print(f"Error loading company financials: {str(e)}")
        return False

def load_asset_manager_holdings(csv_path, driver):
    """Load asset manager holdings from CSV file into Neo4j."""
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return False
    
    try:
        print(f"Loading asset manager holdings from {csv_path}")
        
        # Read the CSV file
        holdings = []
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                holdings.append(row)
        
        print(f"Found {len(holdings)} holdings in the CSV file")
        
        # Create asset manager nodes and relationships in Neo4j
        with driver.session(database="neo4j") as session:
            # Create constraints if they don't exist
            try:
                session.run("""
                    CREATE CONSTRAINT asset_manager_cik_constraint IF NOT EXISTS
                    FOR (am:AssetManager) REQUIRE am.cik IS UNIQUE
                """)
            except Exception as e:
                print(f"Warning: Could not create constraint: {e}")
            
            # Process each holding
            for holding in holdings:
                try:
                    # Extract holding data
                    manager_cik = holding.get('managerCik')
                    manager_name = holding.get('managerName')
                    company_cusip = holding.get('cusip')
                    company_name = holding.get('companyName')
                    value = holding.get('Value')
                    shares = holding.get('shares')
                    share_value = holding.get('share_value')
                    ticker = holding.get('ticker')
                    
                    # Create asset manager node
                    session.run("""
                        MERGE (am:AssetManager {cik: $cik})
                        ON CREATE SET am.name = $name
                        ON MATCH SET am.name = $name
                        RETURN am
                    """, cik=manager_cik, name=manager_name)
                    
                    # Create company node if it doesn't exist
                    session.run("""
                        MERGE (c:Company {cusip: $cusip})
                        ON CREATE SET c.name = $name, c.ticker = $ticker
                        ON MATCH SET c.name = $name, c.ticker = $ticker
                        RETURN c
                    """, cusip=company_cusip, name=company_name, ticker=ticker)
                    
                    # Create holding relationship
                    session.run("""
                        MATCH (am:AssetManager {cik: $manager_cik})
                        MATCH (c:Company {cusip: $company_cusip})
                        MERGE (am)-[h:HOLDS]->(c)
                        ON CREATE SET h.value = $value, h.shares = $shares, h.share_value = $share_value
                        ON MATCH SET h.value = $value, h.shares = $shares, h.share_value = $share_value
                        RETURN h
                    """, manager_cik=manager_cik, company_cusip=company_cusip, 
                        value=value, shares=shares, share_value=share_value)
                
                except Exception as e:
                    print(f"Error processing holding for manager {holding.get('managerCik')}: {e}")
        
        print(f"Successfully loaded asset manager holdings from {csv_path}")
        return True
        
    except Exception as e:
        print(f"Error loading asset manager holdings: {str(e)}")
        return False

def query_knowledge_graph(query_text, driver, retriever_type="hybrid"):
    """Query the knowledge graph using GraphRAG with specified retriever."""
    try:
        # Initialize embeddings
        embedder = OpenAIEmbeddings()
        
        # Try to create a GraphRAG instance
        try:
            from neo4j_graphrag.generation import GraphRAG
            from neo4j_graphrag.retrievers import VectorCypherRetriever
            from neo4j_graphrag.llm import OpenAILLM
            
            # Initialize the LLM
            llm = OpenAILLM(api_key=OPENAI_API_KEY, model_name="gpt-4")
            
            # Set up retrievers
            retrievers = {}
            
            # Vector-based retriever
            vector_retriever = VectorCypherRetriever(
                driver=driver,
                embedder=embedder,
                vector_index_name="chunkEmbeddings",
                chunk_node_label="Chunk",
                chunk_embedding_property="embeddings",
                top_k=5
            )
            retrievers["vector"] = vector_retriever
            
            # Hybrid retriever (using just the vector retriever for now)
            hybrid_retriever = vector_retriever
            retrievers["hybrid"] = hybrid_retriever
            
            # Select the appropriate retriever
            if retriever_type not in retrievers:
                print(f"Warning: Retriever type '{retriever_type}' not found. Using vector retriever.")
                retriever_type = "vector"
            
            selected_retriever = retrievers[retriever_type]
            print(f"Using {retriever_type} retriever for query")
            
            # Create GraphRAG instance with the correct parameters
            graph_rag = GraphRAG(
                retriever=selected_retriever,
                llm=llm
            )
            
            # Execute the query
            print(f"Executing query: {query_text}")
            response = graph_rag.query(query_text)
            
            print("\nQuery Response:")
            print(response)
            return response
            
        except (ImportError, AttributeError) as e:
            print(f"Error creating GraphRAG instance: {e}")
            print("Falling back to basic Neo4j query...")
            
            # Fall back to a basic Neo4j query
            with driver.session(database="neo4j") as session:
                result = session.run(
                    """
                    MATCH (c:Chunk)-[:PART_OF_DOCUMENT]->(d:Document)
                    WHERE c.content CONTAINS $query_param
                    RETURN d.title AS document, c.content AS content
                    LIMIT 5
                    """,
                    query_param=query_text
                )
                print("\nQuery Results:")
                for record in result:
                    print(f"Document: {record.get('document', 'Unknown')}")
                    print(f"Content: {record.get('content', '')[:200]}...\n")
                return None
        
    except Exception as e:
        print(f"Error querying knowledge graph: {str(e)}")
        return None

def main():
    """Main function to run the script."""
    import argparse
    
    # Get the PDF folder from environment variables
    pdf_dir = os.getenv("PDF_PATH")
    
    parser = argparse.ArgumentParser(description="GraphRAG PDF Processing and Querying Tool")
    parser.add_argument("--pdf", help="Path to a single PDF file to process")
    parser.add_argument("--dir", help="Directory containing PDF files to process")
    parser.add_argument("--query", help="Query the knowledge graph")
    parser.add_argument("--retriever", choices=["vector", "keyword", "graph", "hybrid"], 
                        default="hybrid", help="Retriever type to use for querying (default: hybrid)")
    parser.add_argument("--load", action="store_true", help="Load all PDFs from the default directory")
    parser.add_argument("--load-companies", action="store_true", help="Load company financials from CSV")
    parser.add_argument("--load-holdings", action="store_true", help="Load asset manager holdings from CSV")
    parser.add_argument("--use-lexical-graph", action="store_true", help="Use lexical graph builder instead of default")
    
    args = parser.parse_args()
    
    # Connect to Neo4j
    driver = setup_neo4j_connection()
    if not driver:
        return
    
    try:
        if args.query:
            # Query mode
            query_knowledge_graph(args.query, driver, args.retriever)
        elif args.pdf:
            # Single PDF processing mode
            process_pdf(args.pdf, driver, llm, OpenAIEmbeddings(), args.use_lexical_graph)
        elif args.dir:
            # Directory processing mode
            process_multiple_pdfs(args.dir, driver, llm, OpenAIEmbeddings(), args.use_lexical_graph)
        elif args.load_companies:
            # Load company financials
            company_csv = Path("data/cik-10k-urls_short_list.csv")
            if not company_csv.exists():
                print(f"Error: Company CSV file not found at {company_csv}")
                return
            load_company_financials(str(company_csv), driver)
        elif args.load_holdings:
            # Load asset manager holdings
            holdings_csv = Path("data/Asset_Manager_Holdings.csv")
            if not holdings_csv.exists():
                print(f"Error: Holdings CSV file not found at {holdings_csv}")
                return
            load_asset_manager_holdings(str(holdings_csv), driver)
        elif args.load or not (args.query or args.pdf or args.dir or args.load_companies or args.load_holdings):
            # Use the PDF_PATH environment variable if available
            if pdf_dir:
                data_dir = Path(pdf_dir)
            else:
                data_dir = Path("data/form10k-clean_short")
                
            if not data_dir.exists():
                print(f"Error: PDF directory not found at {data_dir}")
                print("Please provide a PDF path using --pdf or a directory using --dir")
                print("Or set the PDF_PATH environment variable in .env file")
                return
            
            print(f"Loading all PDFs from {data_dir}")
            process_multiple_pdfs(str(data_dir), driver, llm, OpenAIEmbeddings(), args.use_lexical_graph)
    finally:
        if driver:
            driver.close()

if __name__ == "__main__":
    main()
