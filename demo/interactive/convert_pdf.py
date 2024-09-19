# %%
from pypdf import PdfReader
import glob
import csv

DIR_PATH = r"./data/supplementary-data/pdf-resumes/\*.pdf"
OUT_PATH = "./data/supplementary-data/pdf-resumes.csv"

# %%
pdfs = glob.glob(DIR_PATH)
id = 0

with open(OUT_PATH, mode="w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    
    writer.writerow(["ID", "Resume"])

    for pdf_path in pdfs:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        writer.writerow([id, text])
        id += 1

# %%
