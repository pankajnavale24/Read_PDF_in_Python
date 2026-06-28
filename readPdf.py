from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path):
    """
    Load a PDF file and return its content as a list of pages.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        list: A list of pages from the PDF file.
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    return pages

print(load_pdf("./The Future of Generative AI.pdf"))