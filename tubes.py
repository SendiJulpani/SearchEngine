import sys
import os
import PyPDF2
from docx import Document
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.uic import loadUi
import math

# Download sumber daya NLTK yang diperlukan

class ProjectBesar(QMainWindow):
    def __init__(self):
        super(ProjectBesar, self).__init__()
        loadUi('gui.ui', self)
        self.pushButton.clicked.connect(self.search)
        self.comboBox.currentIndexChanged.connect(self.load_selected_file)
        self.query_text = ""
        self.documents_directory = 'D:\\#Perkuliahan\\#Semester 5\\IFB-307 DATA MINING DAN INFORMATION RETRIEVAL\\Biggest Project\\Files'
        self.files = []
        self.documents = []
        self.tokenized_documents = []
        self.selected_file_index = -1
        self.pushButton_2.clicked.connect(self.delete_document)
        self.load_files()

    def load_files(self):
        self.files = [f for f in os.listdir(self.documents_directory) if os.path.isfile(os.path.join(self.documents_directory, f))]
        self.comboBox.addItems(self.files)

        # Memuat dan memproses semua dokumen
        self.documents = []
        self.tokenized_documents = []
        for file_name in self.files:
            file_path = os.path.join(self.documents_directory, file_name)
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()

            if ext == '.pdf':
                text = self.read_pdf(file_path)
            elif ext == '.docx':
                text = self.read_docx(file_path)
            else:
                text = self.read_txt(file_path)

            self.documents.append(text)
            self.tokenized_documents.append(self.preprocess_text(text))

    def read_pdf(self, file_path):
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
            return text
    
    def read_docx(self, file_path):
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + " "
        return text
    
    def read_txt(self, file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read()

    def load_selected_file(self):
        self.selected_file_index = self.comboBox.currentIndex()
        if self.selected_file_index != -1:
            file_path = os.path.join(self.documents_directory, self.files[self.selected_file_index])
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()

            if ext == '.pdf':
                text = self.read_pdf(file_path)
            elif ext == '.docx':
                text = self.read_docx(file_path)
            else:
                text = self.read_txt(file_path)

            self.textBrowser.setText(text)

    def preprocess_text(self, text):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()

        stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
        text = stopword_remover.remove(text)
        tokens = nltk.word_tokenize(text)
        tokens = [word.lower() for word in tokens if word.isalnum()]
        tokens = [stemmer.stem(word) for word in tokens]

        return tokens

    def calculate_idf(self, term):
        doc_freq = sum(1 for doc_tokens in self.tokenized_documents if term in doc_tokens)
        return math.log((len(self.tokenized_documents) - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

    def calculate_bm25_score(self, query_tokens, doc_tokens):
        score = 0
        k1 = 1.5
        b = 0.75
        avg_doc_length = sum(len(doc) for doc in self.tokenized_documents) / len(self.tokenized_documents)
        doc_length = len(doc_tokens)

        for term in query_tokens:
            if term in doc_tokens:
                df = sum(1 for doc_tokens in self.tokenized_documents if term in doc_tokens)
                idf = self.calculate_idf(term)
                tf = doc_tokens.count(term)
                score += (idf * tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length)))

        return score

    def rank_documents_bm25(self, query_tokens):
        scores = [(i, self.calculate_bm25_score(query_tokens, doc_tokens)) for i, doc_tokens in enumerate(self.tokenized_documents)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def perform_search(self):
        if self.selected_file_index != -1:
            query = self.query_text
            tokenized_query = self.preprocess_text(query)
            selected_document = self.files[self.selected_file_index]

        # Cari similarity score untuk dokumen yang dipilih
        similarity_score = self.calculate_bm25_score(tokenized_query, self.tokenized_documents[self.selected_file_index])

        # Tampilkan hasil pencarian
        search_results = f"\nBM25 - Hasil Pencarian untuk Dokumen '{selected_document}':\n"

        if similarity_score is not None:
            search_results += f"Similarity Score: {similarity_score:.4f}\n"
        else:
            search_results += "Dokumen tidak ditemukan dalam hasil peringkat.\n"

        # Menyiapkan nilai similarity untuk ditampilkan
        similarity_text = f"\nSimilarity Score untuk Dokumen '{selected_document}':\n"

        if similarity_score is not None:
            similarity_text += f"Similarity Score: {similarity_score:.4f}\n"
        else:
            similarity_text += "Dokumen tidak ditemukan dalam hasil peringkat.\n"

        # Menampilkan nilai similarity di textBrowser_2 tanpa mengganggu tampilan lain
        current_text = self.textBrowser_2.toPlainText()
        self.textBrowser_2.setText(current_text + similarity_text)

        # Menampilkan hasil case folding
        selected_document_text = self.documents[self.selected_file_index]
        case_folded_text = [word.lower() for word in nltk.word_tokenize(selected_document_text) if word.isalnum()]
        self.textBrowser_3.setText(" ".join(case_folded_text))

        # Menampilkan hasil tokenization
        tokenized_text = [word for word in nltk.word_tokenize(selected_document_text) if word.isalnum()]
        self.textBrowser_4.setText(" ".join(tokenized_text))

        # Menampilkan hasil stemming
        stemmed_words_count = self.display_stemmed_words_and_count(selected_document_text)
        self.textBrowser_5.setText(stemmed_words_count)

        preprocessed_text = self.preprocess_text(selected_document_text)
        self.textBrowser_6.setText(" ".join(preprocessed_text))

    def display_stemmed_words_and_count(self, document):
        words = self.preprocess_text(document)
        word_count = {word: words.count(word) for word in set(words)}

        stemmed_results = "Jumlah Kata yang di Stemming : \n"
        for word, count in word_count.items():
            stemmed_results += f"{word}: {count}\n"

        return stemmed_results if stemmed_results else "Tidak ada kata yang diproses\n"

    def search(self):
        self.query_text = self.textEdit.toPlainText()
        self.perform_search()

    def delete_document(self):
         if self.selected_file_index != -1:
            selected_document = self.files[self.selected_file_index]

            # Bersihkan tampilanda
            self.textBrowser_2.clear()
            self.textBrowser_3.clear()
            self.textBrowser_4.clear()
            self.textBrowser_5.clear()
            self.textBrowser_6.clear()

            # Reset indeks dokumen terpilih
            self.selected_file_index = -1

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ProjectBesar()
    window.show()
    sys.exit(app.exec_())
