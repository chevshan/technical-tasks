# Импортируем необходимые для ТЗ библиотеки
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Загружаем данные
def load_data(file):
    data = pd.read_csv(file)
    #  Мы проверяем наличие колонки 'text' в загруженных данных, так как в этом ТЗ
    #  предполагается, что все документы находятся именно в этом столбце.
    #  Также это может предотвратить ошибки, связанные с чтением несуществующих данных
    - Это предотвращает ошибки, связанные с попыткой доступа к несуществующим данным.
    if 'text' in data.columns:
        return data
    return None


# Функция для поиска документов, релевантных запросу пользователя.
# Для преобразования текста в числовой формат использовал TF-IDF. Данный метод учитывает как частоту термина в документе, так и его 
# уникальность по сравнению с другими документами, что помогает лучше выявлять ключевые слова.
def search_documents(query, documents, vectorizer):
    query_vec = vectorizer.transform([query])
    doc_vectors = vectorizer.transform(documents)
    # Использование cosine_similarity позволяет нам вычислить похожесть каждого документа на запрос, что
    # даёт ранжирование документов по их релевантности.
    similarities = cosine_similarity(query_vec, doc_vectors).flatten()
    results = pd.DataFrame({
        'text': documents,
        'similarity': similarities
    }).sort_values(by='similarity', ascending=False)
    # Весь процесс можно описать тремя пунктами:
    # 1. Преобразуем запрос пользователя в векторное представление
    # 2. Преобразуем все документы в векторы, чтобы можно было сравнивать их с запросом.
    # 3. Вычисляем косинусное сходство, после сортируем результат по убыванию
    return results

# Так как в ТЗ сказано что можно использовать готовое, я подумалЮ что можно и визуальный интерфейс взять готовым, потому что делать его с нуля для ТЗ на стажировку как-то чересчур :)
# Поэтому для решения этой проблемы была использована библиотека streamlit (скринштоты результата будут представлены в github)
st.title("Алгоритм поиска информации по документам")
st.write("Загрузите CSV файл с документами:")

uploaded_file = st.file_uploader("Загрузить CSV файл", type="csv")

# После загрузки файла и ввода запроса векторизатор выдаст результат и представит первые 10 лучших результатов по релевантности 
if uploaded_file:
    data = load_data(uploaded_file)
    if data is not None:
        st.write("Первые строки загруженного файла:")
        st.write(data.head())
        
        vectorizer = TfidfVectorizer(stop_words='english')
        vectorizer.fit(data['text'].fillna(''))
        
        query = st.text_input("Введите поисковый запрос:")
        
        if query:
            results = search_documents(query, data['text'].fillna(''), vectorizer)
            
            st.write("Результаты поиска (по убыванию релевантности):")
            st.write(results.head(10))
