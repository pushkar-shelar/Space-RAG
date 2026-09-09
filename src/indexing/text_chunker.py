from pathlib import Path


def chunk_text(
    text,
    chunk_size=1000,
    chunk_overlap=200
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():

            chunks.append(chunk.strip())

        start = end - chunk_overlap

    return chunks


def chunk_document(
    text_dir="data/processed/text"
):

    text_dir = Path(text_dir)

    all_chunks = []

    text_files = sorted(
        text_dir.glob("*.txt")
    )

    for text_file in text_files:

        with open(
            text_file,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()

        chunks = chunk_text(text)

        for chunk_number, chunk in enumerate(
            chunks,
            start=1
        ):

            all_chunks.append(
                {
                    "document": text_file.name.split(
                        "_page_"
                    )[0],
                    "page_number": int(
                        text_file.stem.split("_")[-1]
                    ),
                    "chunk_id": chunk_number,
                    "text": chunk,
                    "source": str(text_file),
                    "modality": "text"
                }
            )

    return all_chunks


if __name__ == "__main__":

    chunks = chunk_document()

    print("\n")
    print("=" * 60)
    print("TEXT CHUNKING")
    print("=" * 60)

    print(
        f"Total chunks: {len(chunks)}"
    )

    for chunk in chunks[:5]:

        print("\n" + "-" * 60)

        print(
            f"Document: {chunk['document']}"
        )

        print(
            f"Page: {chunk['page_number']}"
        )

        print(
            f"Chunk: {chunk['chunk_id']}"
        )

        print(
            f"Characters: {len(chunk['text'])}"
        )

        print(
            chunk["text"][:300]
        )