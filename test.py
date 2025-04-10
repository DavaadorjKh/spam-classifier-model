import pandas as pd

# Өгөгдлийн жишээ (хамгийн сүүлийн багана нь спам эсэхийг илэрхийлнэ)
documents = [
    "Энэ нь спам имэйл",
    "Миний имэйл хаяг",
    "Танд шинэ мэдээлэл байна"
]

# Токенжуулалт болон TF-IDF тооцоолол
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words=['гэх мэт', 'болон', 'ба'])  # Кирилл stop words
X = vectorizer.fit_transform(documents)

# Тухайн текстийн нэрс буюу feature names
feature_names = vectorizer.get_feature_names_out()

# Өгөгдлийг DataFrame болгон хувиргах
df = pd.DataFrame(X.toarray(), columns=feature_names)

# Сүүлийн баганад label (спам эсэх) нэмэх
# 0: спам биш, 1: спам
df['is_spam'] = [1, 0, 0]  # Жишээ өгөгдөл

# Эхний хэдэн мөрийг шалгах
print(df)
