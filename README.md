Retrieval-augmented chatbot experiments

Build:
docker build -t chatbot .

Run:
docker run -p 8000:80 -v .:/app --gpus all chatbot