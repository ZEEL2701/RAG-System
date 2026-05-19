import argparse
import os
from .app import SessionBasedRAG

def run_interactive_mode():
    rag = SessionBasedRAG()
    if not rag.initialize():
        print("Failed to initialize RAG system. Check configuration.")
        return

    print("\n=== Session-Based RAG Interactive Mode ===")
    print("Index documents first. Type 'done' when finished.")

    indexed_files = []
    while True:
        file_path = input("\nEnter document path (or 'done' to continue): ")
        if file_path.lower() == 'done':
            break
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        print(f"Indexing {file_path}...")
        if rag.index_document(file_path):
            print(f"Successfully indexed: {file_path}")
            indexed_files.append(file_path)
        else:
            print(f"Failed to index: {file_path}")

    if not indexed_files:
        print("No documents indexed. Exiting.")
        return

    print("\nAsk questions now! Type 'exit' or 'quit' to stop.")
    while True:
        question = input("\nYour question: ")
        if question.lower() in ['exit', 'quit']:
            break
        print("\nSearching...")
        result = rag.query(question)
        print("\nAnswer:")
        print(result["answer"])
        if result["sources"]:
            print("\nSources:")
            for i, source in enumerate(result["sources"], 1):
                print(f"{i}. {source.get('filename', 'Unknown')} ({source.get('file_type', 'file')})")

    print("\nGoodbye!")

def main():
    parser = argparse.ArgumentParser(description="Session-Based RAG System")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("setup", help="Create sample .env file")

    index_parser = subparsers.add_parser("index", help="Index a document")
    index_parser.add_argument("file_path", type=str, help="Path to document")

    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument("question", type=str, help="Query question")

    subparsers.add_parser("interactive", help="Start interactive mode")

    subparsers.add_parser("serve", help="Launch the Gradio web UI")

    file_parser = subparsers.add_parser("files", help="List/manage files")
    file_parser.add_argument("--list", action="store_true", help="List files")
    file_parser.add_argument("--download", type=int, help="Download file by ID")
    file_parser.add_argument("--output", type=str, help="Download output path")
    file_parser.add_argument("--delete", type=int, help="Delete file by ID")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "setup":
        if not os.path.exists(".env"):
            with open(".env", "w") as f:
                f.write("""# Database config
DB_HOST=localhost
DB_PORT=5432
DB_NAME=Your database name
DB_USER=Your username
DB_PASSWORD=your_password

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text

# Document processing
CHUNK_SIZE=500
CHUNK_OVERLAP=50
COLLECTION_NAME=rag-pgvector

# RAG (lower = smaller Groq prompts; helps free-tier TPM limits)
MAX_CONTEXT_DOCUMENTS=3
SEARCH_K=5
MAX_CHARS_PER_CONTEXT_DOC=900

# Groq
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.1-8b-instant
GROQ_COMPLETION_MAX_TOKENS=512
MAX_TOKENS=256

# AWS S3
S3_ENABLED=False
S3_BUCKET_NAME=your-bucket
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=your region
""")
            print("Sample .env file created. Edit it before running.")
        else:
            print(".env file already exists.")
        return

    if args.command == "serve":
        from .ui import run_gradio_app
        run_gradio_app()
        return

    rag = SessionBasedRAG()
    if not rag.initialize():
        print("Failed to initialize RAG system.")
        return

    if args.command == "index":
        if rag.index_document(args.file_path):
            print(f"Indexed: {args.file_path}")
        else:
            print("Indexing failed.")

    elif args.command == "query":
        result = rag.query(args.question)
        print("\nAnswer:\n", result["answer"])
        if result["sources"]:
            print("\nSources:")
            for i, source in enumerate(result["sources"], 1):
                print(f"{i}. {source.get('filename', 'Unknown')} ({source.get('file_type', 'file')})")

    elif args.command == "files":
        if args.list:
            files = rag.list_files()
            if not files:
                print("No files found.")
            else:
                print(f"{'ID':<5} {'Name':<30} {'Type':<10} {'Uploaded':<20}")
                print("-" * 65)
                for f in files:
                    print(f"{f['file_id']:<5} {f['file_name']:<30} {f['file_type']:<10} {f['upload_date']:<20}")
        elif args.download and args.output:
            if rag.download_file(args.download, args.output):
                print(f"Downloaded to {args.output}")
            else:
                print("Download failed.")
        elif args.delete:
            if rag.file_manager.delete_file(args.delete):
                print(f"Deleted file ID {args.delete}")
            else:
                print("Delete failed.")
        else:
            print("Use --list or --download/--output or --delete")

    elif args.command == "interactive":
        run_interactive_mode()

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
