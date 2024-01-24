# model.py

def process_data(df, query_text):

    import requests
    from bs4 import BeautifulSoup
    import bs4
    import re

    class pat_text:

        def __init__(self, url: str) -> None:
            self.url = url

        def get_soup(self) -> bs4.BeautifulSoup:
            res = requests.get(self.url)
            soup = BeautifulSoup(res.content, 'html.parser')
            self.soup = soup
            return soup

        def get_title(self) -> str:
            title = self.soup.find("meta", attrs={'name': 'DC.title'})["content"]
            return title

        def get_abstract(self) -> str:
            abstract = self.soup.find("div", {"class": "abstract"}).get_text()
            return abstract

        def get_claims(self) -> list:
            claims_list = []
            claims = self.soup.find_all("div", attrs={'class': re.compile('claim-text.*')})
            for clm in claims:
                claims_list.append(clm.get_text())
            return claims_list

        def get_description(self) -> list:
            desc_list = []
            descs = self.soup.find_all("div", attrs={'class': re.compile('description.*')})
            for desc in descs:
                desc_list.append(desc.get_text())
            return desc_list


    import pandas as pd
    import io

    # 新しい列 '処理後の番号' を追加
    # df['処理後の番号'] = df['文献番号'].apply(lambda x: 'JP' + x[2:].replace('-', '').replace('/', ''))
    # df['処理後の番号'] = df['文献番号'].apply(lambda x: x if 'WO' in x else 'JP' + x[2:].replace('-', '').replace('/', ''))
    def format_document_number(x):
        if 'WO' in x:
            return x.replace('/', '')
        else:
            return 'JP' + x[2:].replace('-', '').replace('/', '')

    df['処理後の番号'] = df['文献番号'].apply(format_document_number)

    for index, row in df.iterrows():
        patent_number = row['処理後の番号']
        url = f'https://patents.google.com/patent/{patent_number}/ja'
        patt = pat_text(url)
        patt.get_soup()

        # abstractが存在する場合のみ取得
        abstract_elements = patt.soup.find_all("div", {"class": "abstract"})
        abstract = ''

        if abstract_elements and abstract_elements[0].get_text().strip():
            abstract = patt.get_abstract()

        df.at[index, '要約'] = str(abstract)

        claims = patt.get_claims()
        if claims:
            df.at[index, '請求項1'] = claims[0]
        else:
            df.at[index, '請求項1'] = ""


    # '要約' 列の改行を取り除く
    df['要約'] = df['要約'].apply(lambda x: re.sub(r'\n', '', x) if isinstance(x, str) else x)

    # '請求項1' 列の改行を取り除く
    df['請求項1'] = df['請求項1'].apply(lambda x: re.sub(r'\n', '', x) if isinstance(x, str) else x)

    # '要約' 列の空白文字を取り除く
    df['要約'] = df['要約'].apply(lambda x: x.replace(' ', '') if isinstance(x, str) else x)

    # '請求項1' 列の空白文字を取り除く
    df['請求項1'] = df['請求項1'].apply(lambda x: x.replace(' ', '') if isinstance(x, str) else x)


    # トークナイザとモデルの準備
    from transformers import AutoModel, AutoTokenizer

    # Hugging Face Hubにアップロードされた
    # 教師なしSimCSEのトークナイザとエンコーダを読み込む
    model_name = "llm-book/bert-base-japanese-v3-unsup-simcse-jawiki"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)

    device = "cpu"
    encoder = encoder.to(device)

    import numpy as np
    import torch
    import torch.nn.functional as F

    def embed_texts(texts: list[str]) -> np.ndarray:
        """SimCSEのモデルを用いてテキストの埋め込みを計算"""
        # テキストにトークナイザを適用
        tokenized_texts = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)

        # トークナイズされたテキストをベクトルに変換
        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                encoded_texts = encoder(
                    **tokenized_texts
                ).last_hidden_state[:, 0]

        # ベクトルをNumPyのarrayに変換
        emb = encoded_texts.cpu().numpy().astype(np.float32)
        # ベクトルのノルムが1になるように正規化
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb

    # 要約が無い場合に空欄を文字列として認識させるための処理
    df['要約'] = df['要約'].astype(str)

    from transformers import AutoTokenizer
    from datasets import Dataset

    # テキストデータの前処理やトークナイゼーションに使用するTokenizerを選択
    tokenizer = AutoTokenizer.from_pretrained("llm-book/bert-base-japanese-v3-unsup-simcse-jawiki")

    # データフレームからデータセットに変換
    def tokenize_function(examples):
        return tokenizer(examples["要約"], padding="max_length", truncation=True)

    tokenized_datasets = Dataset.from_pandas(df)
    tokenized_datasets = tokenized_datasets.map(tokenize_function, batched=True)

    # 段落データのすべての事例に埋め込みを付与する
    paragraph_dataset = tokenized_datasets.map(
        lambda examples: {
            "embeddings": list(embed_texts(examples["要約"]))
        },
        batched=True,
    )

    # Faiss による最近傍探索を試す
    import faiss

    # ベクトルの次元数をエンコーダの設定値から取り出す
    emb_dim = encoder.config.hidden_size
    # ベクトルの次元数を指定して空のFaissインデックスを作成する
    index = faiss.IndexFlatIP(emb_dim)
    # 段落データの"embeddings"フィールドのベクトルからFaissインデックスを構築する
    paragraph_dataset.add_faiss_index("embeddings", custom_index=index)

    # query_text = "光重合性化合物と、ポリアルキレンオキサイド基を有しかつメルカプト脂肪酸残基を末端に持つ化合物により表面修飾され、インジウムとリンとを含むコアを持つ量子ドットとを含む波長変換層形成用インクジェットインク。"

    # 最初に元のデータフレームをコピーしておく
    result_df = df.copy()

    # 最近傍探索を実行し、類似度順でなく元の順序で事例とスコアを取得する
    scores, retrieved_examples = paragraph_dataset.get_nearest_examples(
        "embeddings", embed_texts([query_text])[0], k=100
    )

    # 取得した事例の内容をデータフレームに格納
    result_df["文献番号"] = retrieved_examples["文献番号"]
    result_df["要約"] = retrieved_examples["要約"]

    # 元のデータフレームに類似度を含む列を追加
    # result_df["類似度"] = scores
    result_df["類似度"] = [round(score, 4) for score in scores]

    # 元のデータフレームに対して文献番号をキーにしてマージ
    result_df = pd.merge(df, result_df[["文献番号", "類似度"]], on="文献番号", how="left")

    # '要約' 列が空欄の行の '類似度' 列を 0 に設定
    result_df.loc[result_df['要約'] == '', '類似度'] = 0

    # 類似度でデータフレームを並び替え
    result_df = result_df.sort_values(by="類似度", ascending=False)

    return result_df
