from langchain_text_splitters import RecursiveCharacterTextSplitter
from readPdf import load_pdf

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 10,
    chunk_overlap=2
)

pages = load_pdf("./The Future of Generative AI.pdf")
splits = text_splitter.split_documents(pages)

# for i in range(len(splits)):
#     print(splits[i].page_content, end=" ")