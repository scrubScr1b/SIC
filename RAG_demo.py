
import os
import glob
import json
import hashlib
import pandas as pd
import streamlit as st
import re

# from ollama import chat

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings


# =========================================================
# 1. CONFIGURATION
# =========================================================

# st.set_page_config(
#     page_title="ORBIT - Admin Dokumen",
#     page_icon="📁",
#     layout="wide"
# )

# DATA_DIR = "Data"
# CHROMA_DIR = "chroma_db"

# LLM_MODEL = "llama3.2:latest"
# EMBEDDING_MODEL = "nomic-embed-text"

# COLLECTION_NAME = "ORBIT_admin_documents"

# # CONFIGURATION

# CHUNK_SIZE = 1000
# CHUNK_OVERLAP = 150

# # CHUNK_SIZE = 2500
# # CHUNK_OVERLAP = 300

# =========================================================
# 1. CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ORBIT",
    page_icon="📁",
    layout="wide"
)

DATA_DIR = "Data"
CHROMA_DIR = "chroma_db"

COLLECTION_NAME = "ORBIT_admin_documents"

# KONEKTIKA CONFIGURATION
LLM_MODEL = "konektika-thinking"
KONEKTIKA_BASE_URL = "https://konektika.web.id/v1"

# api_key = st.secrets.get(
#     "KONEKTIKA_API_KEY",
#     os.getenv("KONEKTIKA_API_KEY")
# )

# Ambil API Key dari Streamlit Secrets atau Environment Variable
try:
    api_key = st.secrets["KONEKTIKA_API_KEY"]
except (KeyError, FileNotFoundError):
    api_key = os.getenv("KONEKTIKA_API_KEY")

if not api_key:
    st.error("KONEKTIKA_API_KEY belum dikonfigurasi.")
    st.stop()

client = OpenAI(
    api_key=api_key,
    base_url="https://konektika.web.id/v1"
)

if not api_key:
    st.error("KONEKTIKA_API_KEY belum dikonfigurasi.")
    st.stop()

client = OpenAI(
    api_key=api_key,
    base_url=KONEKTIKA_BASE_URL
)

# EMBEDDING CONFIGURATION
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# =========================================================
# 2. KONEKTIKA LLM FUNCTIONS
# =========================================================

def ask_konektika(messages, response_format=None):

    params = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": False
    }

    if response_format:
        params["response_format"] = response_format

    response = client.chat.completions.create(**params)

    return response.choices[0].message.content or ""


def stream_konektika(messages):

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        stream=True
    )

    for chunk in response:

        if not chunk.choices:
            continue

        content = chunk.choices[0].delta.content

        if content:
            yield content

# =========================================================
# 2. LOAD PDF AND EXCEL FILES
# =========================================================

@st.cache_data(show_spinner=False)
def load_all_data():

    pdf_docs = []
    excel_data = {}

    if not os.path.exists(DATA_DIR):
        return pdf_docs, excel_data, [], []

    files = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if os.path.isfile(os.path.join(DATA_DIR, f))
    ]

    pdf_files = [
        f for f in files
        if f.lower().endswith(".pdf")
    ]

    excel_files = [
        f for f in files
        if f.lower().endswith((".xlsx", ".xls"))
        and not os.path.basename(f).startswith("~$")
    ]

    # -----------------------------
    # LOAD PDF
    # -----------------------------

    for file_path in pdf_files:

        try:
            loader = PyPDFLoader(file_path)
            loaded_docs = loader.load()

            for doc in loaded_docs:
                doc.metadata["file_type"] = "PDF"
                doc.metadata["source"] = os.path.basename(file_path)

            pdf_docs.extend(loaded_docs)

        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")

    # -----------------------------
    # LOAD EXCEL
    # -----------------------------

    for file_path in excel_files:

        try:
            sheets = pd.read_excel(
                file_path,
                sheet_name=None,
                header=1,          # Header sebenarnya di baris kedua
                dtype=object
            )

            file_name = os.path.basename(file_path)

            for sheet_name, df in sheets.items():

                df = df.dropna(how="all")
                df = df.dropna(axis=1, how="all")

                # Bersihkan nama kolom
                df.columns = [
                    str(col).strip()
                    for col in df.columns
                ]

                # Hilangkan spasi berlebih
                df.columns = [
                    " ".join(col.split())
                    for col in df.columns
                ]

                # Normalisasi nomor PO
                if "No PO" in df.columns:
                    df["No PO"] = (
                        df["No PO"]
                        .astype("string")
                        .str.replace(r"\.0$", "", regex=True)
                        .str.strip()
                    )

                # Konversi nilai uang
                money_columns = [
                    "MATERIAL",
                    "LABOUR",
                    "TOTAL"
                ]

                for col in money_columns:

                    if col in df.columns:

                        df[col] = pd.to_numeric(
                            df[col]
                            .astype("string")
                            .str.replace(".", "", regex=False)
                            .str.replace(",", ".", regex=False)
                            .str.replace(r"[^\d.-]", "", regex=True),
                            errors="coerce"
                        )

                dataset_key = f"{file_name} | {sheet_name}"

                excel_data[dataset_key] = df

        except Exception as e:
            print(f"Error loading Excel {file_path}: {e}")

    return pdf_docs, excel_data, pdf_files, excel_files


# =========================================================
# 3. CONVERT EXCEL ROWS INTO RAG DOCUMENTS
# =========================================================

def excel_to_documents(excel_data):

    documents = []

    for dataset_key, df in excel_data.items():

        file_name, sheet_name = dataset_key.split(" | ", 1)

        for index, row in df.iterrows():

            row_content = []

            for column, value in row.items():

                if pd.notna(value):
                    row_content.append(
                        f"{column}: {value}"
                    )

            if not row_content:
                continue

            content = (
                f"File Excel: {file_name}\n"
                f"Sheet: {sheet_name}\n"
                f"Baris Excel: {index + 2}\n"
                + "\n".join(row_content)
            )

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": file_name,
                        "file_type": "Excel",
                        "sheet": str(sheet_name),
                        "row": int(index + 2)
                    }
                )
            )

    return documents


# =========================================================
# 4. CREATE DOCUMENT CHUNKS
# =========================================================

def create_chunks(pdf_docs, excel_docs):

    all_docs = pdf_docs + excel_docs

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(all_docs)

    return chunks


# =========================================================
# 5. CREATE / UPDATE CHROMA DATABASE
# =========================================================

# @st.cache_resource(show_spinner=False)
# def get_vectorstore():

#     embeddings = OllamaEmbeddings(
#         model=EMBEDDING_MODEL
#     )

#     vectorstore = Chroma(
#         collection_name=COLLECTION_NAME,
#         embedding_function=embeddings,
#         persist_directory=CHROMA_DIR
#     )

#     return vectorstore

@st.cache_resource(show_spinner=False)
def get_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

    return vectorstore


def create_document_id(doc):

    content = (
        json.dumps(
            doc.metadata,
            sort_keys=True,
            ensure_ascii=False
        )
        + doc.page_content
    )

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


def index_documents(vectorstore, chunks):

    if not chunks:
        return 0

    existing_ids = set(
        vectorstore.get()["ids"]
    )

    new_docs = []
    new_ids = []

    for doc in chunks:

        doc_id = create_document_id(doc)

        if doc_id not in existing_ids:

            new_docs.append(doc)
            new_ids.append(doc_id)

            existing_ids.add(doc_id)

    if new_docs:

        vectorstore.add_documents(
            documents=new_docs,
            ids=new_ids
        )

    return len(new_docs)


# =========================================================
# 6. RAG RETRIEVAL
# =========================================================

def retrieve_documents(vectorstore, query, k=15):
    
    retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": k,
        "fetch_k": 60,
        "lambda_mult": 0.7
        }
    )

    results = retriever.invoke(query)

    context = []

    for doc in results:

        metadata = doc.metadata

        source = metadata.get("source", "Unknown")
        file_type = metadata.get("file_type", "PDF")

        if file_type == "Excel":

            source_info = (
                f"File: {source} | "
                f"Sheet: {metadata.get('sheet')} | "
                f"Baris: {metadata.get('row')}"
            )

        else:

            source_info = (
                f"File: {source} | "
                f"Halaman: {metadata.get('page', '-')}"
            )

        context.append(
            f"{source_info}\n{doc.page_content}"
        )

    return "\n\n".join(context), results

def get_unique_sources(sources):

    unique_sources = []
    seen = set()

    for doc in sources:

        metadata = doc.metadata

        file_type = metadata.get("file_type", "PDF")
        source = metadata.get("source", "Unknown")

        if file_type == "Excel":

            key = (
                "Excel",
                source,
                metadata.get("sheet"),
                metadata.get("row")
            )

        else:

            key = (
                "PDF",
                source,
                metadata.get("page")
            )

        if key not in seen:

            seen.add(key)
            unique_sources.append(metadata)

    return unique_sources

# =========================================================
# CHECKING INDEXED DOCUMENTS
# =========================================================

# def check_indexed_documents(vectorstore):

#     all_data = vectorstore.get(
#         include=["metadatas"]
#     )

#     sources = set()

#     for metadata in all_data["metadatas"]:
#         if metadata:
#             sources.add(metadata.get("source", "Unknown"))

#     result = {
#         "total_chunks": len(all_data["metadatas"]),
#         "total_files": len(sources),
#         "files": sorted(sources)
#     }

#     return result

# vectorstore = get_vectorstore()

# index_info = check_indexed_documents(vectorstore)

# st.sidebar.markdown("### 📚 Database ORBIT")

# st.sidebar.metric(
#     "Total Indexed Files",
#     index_info["total_files"]
# )

# st.sidebar.metric(
#     "Total Indexed Chunks",
#     index_info["total_chunks"]
# )

# with st.sidebar.expander("Lihat daftar dokumen"):
#     for file in index_info["files"]:
#         st.write(file)

# =========================================================
# 4. CREATE ANALYSIS PLAN USING KONEKTIKA
# =========================================================

def create_analysis_plan(question, excel_data):

    dataset_information = []

    for dataset_key, df in excel_data.items():

        sample = df.head(3).fillna("").astype(str)

        dataset_information.append({
            "dataset": dataset_key,
            "columns": list(df.columns),
            "sample_data": sample.to_dict(orient="records")
        })

    prompt = f"""
Anda adalah asisten admin dan data analyst.

Tentukan apakah pertanyaan pengguna membutuhkan:
- excel: analisis menggunakan Pandas
- document: pencarian dokumen menggunakan RAG
- both: membutuhkan Excel dan dokumen

Data Excel yang tersedia:
{json.dumps(dataset_information, ensure_ascii=False, default=str)}

Pertanyaan pengguna:
{question}

Buat rencana operasi Excel jika diperlukan.
Jangan membuat kode Python.

Gunakan JSON dengan format:

{{
    "source": "excel",
    "dataset": "nama dataset yang dipilih",
    "task": "sum",
    "column": "nama kolom",
    "group_by": [],
    "filters": [],
    "sort_by": null,
    "ascending": false,
    "top_n": 10
}}

Pilihan source:
excel, document, both

Pilihan task:
sum, mean, count, count_rows, min, max,
groupby_sum, groupby_mean, groupby_count,
top_n, describe, list

Operator filter:
contains, equals, not_equals, gt, gte, lt, lte

Aturan:
1. Gunakan dataset dan kolom yang tersedia.
2. Total nilai menggunakan sum.
3. Jumlah baris menggunakan count_rows.
4. Rekap kategori menggunakan groupby_sum atau groupby_count.
5. Nilai tertinggi menggunakan top_n.
6. Jika tidak berkaitan dengan Excel, gunakan document.
7. Jika membutuhkan Excel dan PDF, gunakan both.
8. Jangan mengarang nama kolom.
9. Keluarkan JSON valid saja.
"""

    response = ask_konektika(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # Mengantisipasi JSON yang dibungkus markdown
    response = response.strip()

    response = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        response
    )

    return json.loads(response)


# =========================================================
# 8. VALIDATE AND EXECUTE PANDAS PLAN
# =========================================================

def analyze_excel(plan, excel_data):

    dataset_key = plan.get("dataset")

    if dataset_key not in excel_data:
        return None, "Dataset Excel tidak ditemukan."

    df = excel_data[dataset_key].copy()

    # -----------------------------
    # VALIDATE COLUMNS
    # -----------------------------

    valid_columns = list(df.columns)

    def get_valid_column(column):

        if column is None:
            return None

        for valid_column in valid_columns:

            if str(valid_column).lower() == str(column).lower():
                return valid_column

        raise ValueError(
            f"Kolom '{column}' tidak ditemukan."
        )

    # -----------------------------
    # APPLY FILTERS
    # -----------------------------

    for condition in plan.get("filters", []):

        column = get_valid_column(
            condition.get("column")
        )

        operator = condition.get("operator")
        value = condition.get("value")

        if operator not in [
            "contains", "equals", "not_equals",
            "gt", "gte", "lt", "lte"
        ]:
            raise ValueError("Operator filter tidak valid.")

        series = df[column]

        if operator == "contains":

            mask = series.astype(str).str.contains(
                str(value),
                case=False,
                na=False,
                regex=False
            )

        elif operator == "equals":

            mask = (
                series.astype(str).str.lower()
                == str(value).lower()
            )

        elif operator == "not_equals":

            mask = (
                series.astype(str).str.lower()
                != str(value).lower()
            )

        else:

            numeric_series = pd.to_numeric(
                series.astype(str).str.replace(",", "", regex=False),
                errors="coerce"
            )

            numeric_value = pd.to_numeric(
                str(value).replace(",", ""),
                errors="coerce"
            )

            if pd.isna(numeric_value):
                raise ValueError("Nilai filter numerik tidak valid.")

            if operator == "gt":
                mask = numeric_series > numeric_value

            elif operator == "gte":
                mask = numeric_series >= numeric_value

            elif operator == "lt":
                mask = numeric_series < numeric_value

            else:
                mask = numeric_series <= numeric_value

        df = df[mask]

    # -----------------------------
    # EXECUTE TASK
    # -----------------------------

    task = plan.get("task")
    column = get_valid_column(plan.get("column"))

    if task == "count_rows":

        result = pd.DataFrame({
            "Keterangan": ["Jumlah baris"],
            "Hasil": [len(df)]
        })

    elif task == "count":

        if column is None:
            raise ValueError("Kolom untuk count belum ditentukan.")

        result = pd.DataFrame({
            "Keterangan": [f"Jumlah data {column}"],
            "Hasil": [df[column].count()]
        })

    elif task in ["sum", "mean", "min", "max"]:

        if column is None:
            raise ValueError("Kolom analisis belum ditentukan.")

        numeric = pd.to_numeric(
            df[column].astype(str).str.replace(",", "", regex=False),
            errors="coerce"
        )

        if task == "sum":
            value = numeric.sum()

        elif task == "mean":
            value = numeric.mean()

        elif task == "min":
            value = numeric.min()

        else:
            value = numeric.max()

        result = pd.DataFrame({
            "Keterangan": [f"{task.upper()} - {column}"],
            "Hasil": [value]
        })

    elif task in [
        "groupby_sum",
        "groupby_mean",
        "groupby_count"
    ]:

        group_columns = [
            get_valid_column(c)
            for c in plan.get("group_by", [])
        ]

        if not group_columns:
            raise ValueError("Kolom pengelompokan belum ditentukan.")

        if task == "groupby_count":

            result = (
                df.groupby(group_columns, dropna=False)
                .size()
                .reset_index(name="Jumlah Data")
            )

        else:

            if column is None:
                raise ValueError("Kolom nilai belum ditentukan.")

            df[column] = pd.to_numeric(
                df[column].astype(str).str.replace(",", "", regex=False),
                errors="coerce"
            )

            if task == "groupby_sum":

                result = (
                    df.groupby(group_columns, dropna=False)[column]
                    .sum()
                    .reset_index()
                )

            else:

                result = (
                    df.groupby(group_columns, dropna=False)[column]
                    .mean()
                    .reset_index()
                )

    elif task == "top_n":

        if column is None:
            raise ValueError("Kolom ranking belum ditentukan.")

        df[column] = pd.to_numeric(
            df[column].astype(str).str.replace(",", "", regex=False),
            errors="coerce"
        )

        result = df.sort_values(
            by=column,
            ascending=bool(plan.get("ascending", False))
        ).head(
            max(1, min(int(plan.get("top_n", 10)), 100))
        )

    elif task == "describe":

        result = df.describe(
            include="all"
        ).reset_index()

    elif task == "list":

        result = df.head(100)

    else:

        raise ValueError(
            f"Jenis analisis '{task}' belum didukung."
        )

    return result, None


# =========================================================
# 9. GENERATE ANSWER
# =========================================================

# def generate_answer(question, context):

#     prompt = f"""
# Anda adalah ORBIT, asisten admin dokumen.

# Jawab pertanyaan berdasarkan konteks dokumen berikut.

# Aturan:
# - Gunakan bahasa Indonesia.
# - Jawab dengan jelas dan profesional.
# - Jangan mengarang informasi.
# - Jika informasi tidak ditemukan, katakan tidak ditemukan.
# - Sebutkan nama file dan halaman jika tersedia.

# Konteks:
# {context}

# Pertanyaan:
# {question}
# """

#     response = chat(
#         model=LLM_MODEL,
#         messages=[
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ]
#     )
def generate_answer(question, context):

    prompt = f"""
Anda adalah ORBIT, asisten admin dokumen.

Jawab pertanyaan berdasarkan konteks dokumen berikut.

Aturan:
- Gunakan bahasa Indonesia.
- Jawab dengan jelas dan profesional.
- Jangan mengarang informasi.
- Jika informasi tidak ditemukan, katakan tidak ditemukan.
- Sebutkan nama file dan halaman jika tersedia.

Konteks:
{context}

Pertanyaan:
{question}
"""

    return stream_konektika(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.message.content


# def explain_excel_result(question, dataset_key, result):

#     result_text = result.to_string(index=False)

#     prompt = f"""
# Anda adalah ORBIT, asisten admin dan data analyst.

# Jelaskan hasil analisis Excel berikut dalam bahasa Indonesia.

# Pertanyaan:
# {question}

# Sumber data:
# {dataset_key}

# Hasil perhitungan Pandas:
# {result_text}

# Aturan:
# - Gunakan angka hasil perhitungan apa adanya.
# - Jangan mengubah atau mengarang angka.
# - Jelaskan kesimpulan secara singkat.
# - Jika hasil berupa tabel, jelaskan poin pentingnya.
# - Sebutkan bahwa hasil berasal dari file Excel.
# """

#     response = chat(
#         model=LLM_MODEL,
#         messages=[
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ]
#     )
def explain_excel_result(question, dataset_key, result):

    result_text = result.to_string(index=False)

    prompt = f"""
    Anda adalah ORBIT, asisten admin dan data analyst.

    Jelaskan hasil analisis Excel berikut dalam bahasa Indonesia.

    Pertanyaan:
    {question}

    Sumber data:
    {dataset_key}

    Hasil perhitungan Pandas:
    {result_text}

    Aturan:
    - Gunakan angka hasil perhitungan apa adanya.
    - Jangan mengubah atau mengarang angka.
    - Jelaskan kesimpulan secara singkat.
    - Jika hasil berupa tabel, jelaskan poin pentingnya.
    - Sebutkan bahwa hasil berasal dari file Excel.
    """

    return stream_konektika(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.message.content

def normalize_po(value):
    if pd.isna(value):
        return ""

    value = str(value).strip()
    value = re.sub(r"\.0$", "", value)
    value = re.sub(r"\D", "", value)

    return value


def search_po(po_number, excel_data):

    target_po = normalize_po(po_number)
    results = []

    for dataset_key, df in excel_data.items():

        if "No PO" not in df.columns:
            continue

        matched = df[
            df["No PO"].apply(normalize_po) == target_po
        ]

        if not matched.empty:
            result = matched.copy()
            result["Source Excel"] = dataset_key
            results.append(result)

    if not results:
        return None

    return pd.concat(results, ignore_index=True)


# =========================================================
# 10. STREAMLIT INTERFACE
# =========================================================

st.title("🛰️ ORBIT AI AGENT")
st.caption(
    "Operational Records & Business Intelligence Technology - Asisten AI untuk pencarian & analisa"
)

with st.spinner("Membaca dokumen PDF"):

    pdf_docs, excel_data, pdf_files, excel_files = load_all_data()

    excel_docs = excel_to_documents(excel_data)

    chunks = create_chunks(
        pdf_docs,
        excel_docs
    )

    vectorstore = get_vectorstore()

    new_embeddings = index_documents(
        vectorstore,
        chunks
    )

# =========================================================
# 11. SIDEBAR INFORMATION
# =========================================================

st.sidebar.header("📊 Document Information")

st.sidebar.metric(
    "Total PDF",
    len(pdf_files)
)

st.sidebar.metric(
    "Total Excel",
    len(excel_files)
)

st.sidebar.metric(
    "Total PDF Pages",
    len(pdf_docs)
)

st.sidebar.metric(
    "Total Excel Rows",
    sum(len(df) for df in excel_data.values())
)

st.sidebar.metric(
    "Total RAG Chunks",
    len(chunks)
)

st.sidebar.metric(
    "New Embeddings",
    new_embeddings
)

# =========================================================
# 12. CHAT INTERFACE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("table") is not None:
            st.dataframe(
                message["table"],
                use_container_width=True
            )

user_query = st.chat_input(
    "Tanyakan ORBIT...."
)

if user_query:

    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):

        answer = ""
        result_table = None
        answer_displayed = False

        try:
            
            # =====================================
            # DIRECT PO SEARCH USING PANDAS
            # =====================================

            po_match = re.search(
                r"\b\d{10}\b",
                user_query
            )

            if po_match:

                po_number = po_match.group(0)

                po_result = search_po(
                    po_number,
                    excel_data
                )

                if po_result is not None:

                    st.markdown("### 📊 Hasil Pencarian PO")

                    st.dataframe(
                        po_result,
                        use_container_width=True
                    )

                    total_value = po_result["TOTAL"].sum()

                    answer = (
                        f"PO **{po_number}** ditemukan.\n\n"
                        f"**Total nilai PO: "
                        f"Rp {total_value:,.0f}**"
                    )

                    st.success(answer)
                    
                    answer_displayed = True

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                else:

                    answer = f"PO {po_number} tidak ditemukan di Excel."

                    st.warning(answer)

                    answer_displayed = True

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

            else:

                # =====================================
                # LANJUTKAN PROSES AI / RAG YANG LAMA
                # =====================================

                # Letakkan seluruh kode lama mulai dari:
                # if excel_data:
                #
                # sampai proses generate answer
                # di dalam blok else ini.

                # -----------------------------
                # DETERMINE QUERY TYPE
                # -----------------------------

                if excel_data:

                    with st.spinner("Memahami pertanyaan..."):

                        plan = create_analysis_plan(
                            user_query,
                            excel_data
                        )

                    source_type = plan.get(
                        "source",
                        "document"
                    )

                else:

                    source_type = "document"
                    plan = {}

                # -----------------------------
                # EXCEL ANALYSIS
                # -----------------------------

                if source_type in ["excel", "both"]:

                    with st.spinner("Menganalisis data Excel dengan Pandas..."):

                        result_table, error = analyze_excel(
                            plan,
                            excel_data
                        )

                    if error:

                        answer = f"Analisis Excel gagal: {error}"

                    else:

                        # answer = explain_excel_result(
                        #     user_query,
                        #     plan.get("dataset"),
                        #     result_table
                        # )
                        answer = st.write_stream(
                            explain_excel_result(
                                user_query,
                                plan.get("dataset"),
                                result_table
                            )
                        )

                        answer_displayed = True

                        st.markdown("### 📊 Hasil Analisis Excel")

                        st.dataframe(
                            result_table,
                            use_container_width=True
                        )

                        st.caption(
                            f"Sumber: {plan.get('dataset')}"
                        )

                # -----------------------------
                # PDF / RAG SEARCH
                # -----------------------------

                if source_type in ["document", "both"]:

                    with st.spinner("Mencari informasi..."):

                        context, sources = retrieve_documents(
                            vectorstore,
                            user_query,
                            k=15
                        )


                        # document_answer = generate_answer(
                        #     user_query,
                        #     context
                        # )
                        document_answer = st.write_stream(
                            generate_answer(
                                user_query,
                                context
                            )
                        )

                        answer_displayed = True

                    if source_type == "both":

                        answer += (
                            "\n\n### 📄 Informasi dari Dokumen\n\n"
                            + document_answer
                        )

                    else:

                        answer = document_answer

                    # if sources:

                    #     # st.markdown("### 📚 Referensi Dokumen")

                    #     # for source in sources:

                    #     #     metadata = source.metadata

                    #     #     if metadata.get("file_type") == "Excel":

                    #     #         st.caption(
                    #     #             f"Excel: {metadata.get('source')} | "
                    #     #             f"Sheet: {metadata.get('sheet')} | "
                    #     #             f"Baris: {metadata.get('row')}"
                    #     #         )

                    #     #     else:

                    #     #         st.caption(
                    #     #             f"PDF: {metadata.get('source')} | "
                    #     #             f"Halaman: {metadata.get('page', '-')}"
                    #     #         )

                    #     unique_sources = get_unique_sources(sources)

                    #     st.markdown("### 📚 Referensi Dokumen")

                    #     for metadata in unique_sources:

                    #         if metadata.get("file_type") == "Excel":

                    #             st.caption(
                    #                 f"Excel: {metadata.get('source')} | "
                    #                 f"Sheet: {metadata.get('sheet')} | "
                    #                 f"Baris: {metadata.get('row')}"
                    #             )

                    #         else:

                    #             page = metadata.get("page")

                    #             # Nomor halaman PDF dimulai dari 1
                    #             page_number = page + 1 if page is not None else "-"

                    #             st.caption(
                    #                 f"PDF: {metadata.get('source')} | "
                    #                 f"Halaman: {page_number}"
                    #             )

            # -----------------------------
            # DISPLAY ANSWER
            # -----------------------------

            # st.markdown(answer)

            # st.session_state.messages.append({
            #     "role": "assistant",
            #     "content": answer,
            #     "table": (
            #         result_table.to_dict(orient="records")
            #         if result_table is not None
            #         else None
            #     )
            # })
            if not answer_displayed and answer:

                st.markdown(answer)

            # Simpan jawaban satu kali ke chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "table": (
                    result_table.to_dict(orient="records")
                    if result_table is not None
                    else None
                )
            })

        except Exception as e:

            error_message = f"Terjadi kesalahan: {str(e)}"

            st.error(error_message)

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message
            })
