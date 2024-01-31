from fastapi import FastAPI, HTTPException, File, UploadFile, Form
import pandas as pd
from chardet.universaldetector import UniversalDetector
import logging
from typing import Optional
import io
from model import process_data

# ロガーの設定
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# エンコーディング検出関数
def detect_encoding(contents):
    detector = UniversalDetector()
    for line in contents.split(b'\n'):
        detector.feed(line)
        if detector.done:
            break
    detector.close()
    return detector.result['encoding']
    
# トップページ
@app.get('/')
async def index():
    return {"Patents": 'Patents_Similarity_App'}
    
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...), query_text: Optional[str] = Form(None)):
    try:
        contents = await file.read()
        logger.debug(f"Read {len(contents)} bytes from the file.")

        if len(contents) == 0:
            logger.error("No content was read from the file. The file might be empty.")
            raise HTTPException(status_code=400, detail="ファイルが空です。")

        file_encoding = detect_encoding(contents)
        if not file_encoding:
            raise HTTPException(
                status_code=400,
                detail=f"ファイルのエンコーディングを検出できませんでした。"
            )
        decoded_contents = contents.decode(file_encoding)
        if not decoded_contents.strip():
            raise HTTPException(
                status_code=400,
                detail=f"デコードされたファイルの内容が空です。"
            )
        csv_data = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        # csv_data = pd.read_csv(io.StringIO(decoded_contents))
        result_df = process_data(csv_data, query_text)

        # DataFrameをJSON形式でクライアントに返す
        return result_df.to_json(orient='records', force_ascii=False)

    except Exception as e:
        logger.exception(f"An error occurred while processing the file: {e}")
        raise HTTPException(status_code=500, detail="内部サーバーエラー")
