import streamlit as st
import pandas as pd
import requests
import json
import base64
from io import BytesIO

# Display uploaded file
def display_uploaded_file(uploaded_file):
    st.subheader("Uploaded CSV file:")
    try:
        df = pd.read_csv(uploaded_file)
        st.write(df)
    except pd.errors.EmptyDataError:
        st.error("Uploaded CSV file is empty. Please upload a file with data.")

# Streamlitのフォームからデータを送信して処理結果を取得する関数
def get_processed_data(uploaded_file, query_text):
    url = "https://patents-similarity-app.onrender.com/uploadfile/"
    files = {"file": uploaded_file.getvalue()}
    data = {"query_text": query_text}
    response = requests.post(url, files=files, data=data)
    return pd.read_json(response.json(), orient='records')

# Streamlit UI
st.title("Patents Similarity App")

st.markdown(
    """
    このアプリは、公知特許の要約と、入力した文章（あなたの発明の要件）との類似度を計算します。
    公知例調査の効率化にお役立て下さい。
    まずCSVファイル（公知特許の文献番号リスト）と、文章（あなたの発明の要件）を用意してください。
    なお公知特許の要約は、文献番号を用いて自動でスクレイピングして取得します。
    
    **使用方法**<br>
    **1.** J-Plat Pat等の特許検索ソフトを用いてキーワード検索等を行い、発明特許に近い公知文献を抽出します。<br>
    **2.** 文献番号のリストを作成します。CSVファイルの1行目に「文献番号」と記載し、2行目以降に各文献の文献番号（例：特開○○-○○, 特許○○, 再表○○/○○）を1行ずつ記載します。<br>
    　J-Plat PatでCSV出力した国内文献のファイルはそのまま使えます。<br>
    　外国文献についてはWO2000/000000の形式にしてください。<br>
    **3.** CSVファイルをアップロードし、発明特許の要件を文章でテキスト入力欄に記載します。<br>
    **4.** Run Processingボタンを押します。実行完了までに数分を要します。<br>
    **5.** 「Done!」が表示されたら実行完了です。<br>
    **6.** Results欄に元のCSVファイルの内容に、スクレイピングの際に用いた処理後の番号、要約、請求項1、類似度の列が追加されて表示されます。<br>
    **7.** 表にカーソルを合わせると、右上にDownloadボタンが表示されますのでダウンロードして編集できます。<br>
    **8.** 要約が抽出できない文献は要約欄が空欄となり、類似度は0と表示されます。

    """,
    unsafe_allow_html=True
)
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
query_text = st.text_area("Enter text in Japanese (requirements of your patent)", "", height=200)

if st.button("Run Processing"):
    if uploaded_file is not None and query_text:
        with st.spinner('Processing...'):
            display_uploaded_file(uploaded_file)
            df = get_processed_data(uploaded_file, query_text)
        st.success('Done! Similarities are listed in the last column of the table below.')
        st.subheader("Results:")
        st.dataframe(df)  # 処理結果を表示

        # CSVダウンロードリンクを生成する関数
        def get_table_download_link(df):
            csv = df.to_csv(index=False)  # CSV文字列としてデータフレームを変換
            b64 = base64.b64encode(csv.encode()).decode()  # バイナリにエンコードし、その後base64エンコードされた文字列にデコード
            href = f'<a href="data:file/csv;base64,{b64}" download="processed_data.csv">Download CSV file</a>'
            return href

        # ダウンロードリンクを表示
        st.markdown(get_table_download_link(df), unsafe_allow_html=True)
    else:
        st.error('Please upload a CSV file and enter the text.')
