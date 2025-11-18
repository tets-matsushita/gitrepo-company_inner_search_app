"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
import sys
import unicodedata
from dotenv import load_dotenv
import streamlit as st
from docx import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document as LC_Document
# 追加: Webページ読み込み用ローダーの import
from langchain_community.document_loaders import WebBaseLoader
import constants as ct


############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()


############################################################
# 関数定義
############################################################

logger = logging.getLogger(__name__)

def initialize():
    logger.info("initialize() 開始")
    try:
        # 初期化データの用意
        initialize_session_state()
        # ログ出力用にセッションIDを生成
        initialize_session_id()
        # ログ出力の設定
        initialize_logger()
        # RAGのRetrieverを作成
        initialize_retriever()
    except Exception as e:
        logger.exception("initialize() 内で例外発生")
        # 詳細をファイルにも書きたい場合は logger を設定してください
        raise  # main.py 側でキャッチするため再スロー


def initialize_logger():
    """
    ログ出力の設定
    """
    # 指定のログフォルダが存在すれば読み込み、存在しなければ新規作成
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)
    
    # 引数に指定した名前のロガー（ログを記録するオブジェクト）を取得
    # 再度別の箇所で呼び出した場合、すでに同じ名前のロガーが存在していれば読み込む
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにロガーにハンドラー（ログの出力先を制御するもの）が設定されている場合、同じログ出力が複数回行われないよう処理を中断する
    if logger.hasHandlers():
        return

    # 1日単位でログファイルの中身をリセットし、切り替える設定
    log_handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
        when="D",
        encoding="utf8"
    )
    # 出力するログメッセージのフォーマット定義
    # - 「levelname」: ログの重要度（INFO, WARNING, ERRORなど）
    # - 「asctime」: ログのタイムスタンプ（いつ記録されたか）
    # - 「lineno」: ログが出力されたファイルの行番号
    # - 「funcName」: ログが出力された関数名
    # - 「session_id」: セッションID（誰のアプリ操作か分かるように）
    # - 「message」: ログメッセージ
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, session_id={st.session_state.session_id}: %(message)s"
    )

    # 定義したフォーマッターの適用
    log_handler.setFormatter(formatter)

    # ログレベルを「INFO」に設定
    logger.setLevel(logging.INFO)

    # 作成したハンドラー（ログ出力先を制御するオブジェクト）を、
    # ロガー（ログメッセージを実際に生成するオブジェクト）に追加してログ出力の最終設定
    logger.addHandler(log_handler)


def initialize_session_id():
    """
    セッションIDの作成
    """
    if "session_id" not in st.session_state:
        # ランダムな文字列（セッションID）を、ログ出力用に作成
        st.session_state.session_id = uuid4().hex


def initialize_retriever():
    """
    画面読み込み時にRAGのRetriever（ベクターストアから検索するオブジェクト）を作成
    """
    # ロガーを読み込むことで、後続の処理中に発生したエラーなどがログファイルに記録される
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにRetrieverが作成済みの場合、後続の処理を中断
    if "retriever" in st.session_state:
        return
    
    # RAGの参照先となるデータソースの読み込み
    docs_all = load_data_sources()

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])
    
    # 埋め込みモデルの用意
    embeddings = OpenAIEmbeddings()
    
    """
    ★★★問題２の回答★★★
    """
    # チャンク分割用のオブジェクトを作成
    text_splitter = CharacterTextSplitter(
        chunk_size=ct.CHUNK_SIZE,
        chunk_overlap=ct.CHUNK_OVERLAP,
         separator="\n"
    )

    # チャンク分割を実施
    splitted_docs = text_splitter.split_documents(docs_all)

    # --- デバッグ出力: チャンク分割後の件数と先頭サンプル ---
    print("=== split debug ===")
    print(f"splitted_docs 件数: {len(splitted_docs)}")
    for i, d in enumerate(splitted_docs[:5]):
        src = d.metadata.get("source", "<no source>")
        print(f"  [{i}] source={src} preview={d.page_content[:80]!r}")
    print("===================")

    # ベクターストアの作成
    # Chroma.from_documents は embedding_function=... を使う（実装に依存するため明示）
    db = Chroma.from_documents(splitted_docs, embedding_function=embeddings)

    # デバッグ: Chroma に格納された件数（内部コレクションにアクセス）
    try:
        print("Chroma collection count:", db._collection.count())
    except Exception as e:
        print("Chroma count unavailable:", e)

    # ベクターストアを検索するRetrieverの作成
    """
    ★★★問題１の回答★★★
    ★★★問題２の回答★★★
    """
    st.session_state.retriever = db.as_retriever(search_kwargs={"k": ct.RETRIEVER_K})


def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        # 「表示用」の会話ログを順次格納するリストを用意
        st.session_state.messages = []
        # 「LLMとのやりとり用」の会話ログを順次格納するリストを用意
        st.session_state.chat_history = []


def load_data_sources():
    """
    RAGの参照先となるデータソースの読み込み

    Returns:
        読み込んだ通常データソース
    """
    # データソースを格納する用のリスト
    docs_all = []
    # ファイル読み込みの実行（渡した各リストにデータが格納される）
    recursive_file_check(ct.RAG_TOP_FOLDER_PATH, docs_all)

    web_docs_all = []
    # ファイルとは別に、指定のWebページ内のデータも読み込み
    # 読み込み対象のWebページ一覧に対して処理
    for web_url in ct.WEB_URL_LOAD_TARGETS:
        # 指定のWebページを読み込み
        loader = WebBaseLoader(web_url)
        web_docs = loader.load()
        # for文の外のリストに読み込んだデータソースを追加
        web_docs_all.extend(web_docs)
    # 通常読み込みのデータソースにWebページのデータを追加
    docs_all.extend(web_docs_all)

    # （読み込み処理実行後に追加）
    print("=== 読み込み完了デバッグ ===")
    print(f"docs_all 件数: {len(docs_all)}")
    if len(docs_all) > 0:
        print("先頭ドキュメント metadata:", docs_all[0].metadata)
        print("先頭ドキュメント content preview:", docs_all[0].page_content[:200])
    print("=========================")

    return docs_all


def recursive_file_check(path, docs_all):
    """
    フォルダを再帰的に探索して、対応する拡張子を file_load に渡す。
    対象外フォルダ（.db や データベース化済み 等）はスキップする。
    """
    if os.path.isdir(path):
        base = os.path.basename(path)
        if base in ["データベース化済み", ".db"]:

            # スキップ
            return
        for name in os.listdir(path):
            recursive_file_check(os.path.join(path, name), docs_all)
    else:
        # ファイルの場合、拡張子を小文字で判定
        ext = os.path.splitext(path)[1].lower()
        if ext in ct.SUPPORTED_EXTENSIONS:
            print(f"▶ 読み込み対象ファイル: {path}")
            file_load(path, docs_all)
        else:
            # .txt の場合は TextLoader entry があるはずだが、念のため拡張子直接判定も可能
            if ext == ".txt":
                print(f"▶ 読み込み対象（txt）: {path}")
                file_load(path, docs_all)
            else:
                print(f"⊘ スキップ（非対応拡張子）: {path}")


def file_load(path, docs_all):
    """
    単一ファイルを読み込み、docs_all に LangChain の Document 形式で追加する。
    - .txt は encoding を UTF-8 で試し、失敗したら cp932 を fallback。
    - 他は constants.SUPPORTED_EXTENSIONS のローダーを利用する。
    """
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".txt":
            # txt は直接読み込んで Document にする（encoding fallback 対応）
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(path, "r", encoding="cp932", errors="ignore") as f:
                    text = f.read()
            # metadata を他のローダーと揃える（source は / に正規化、page を 0 として付与）
            norm_source = path.replace("\\", "/")
            docs_all.append(LC_Document(page_content=text, metadata={"source": norm_source, "page": 0}))
            print(f"✓ txt を追加しました: {path}")
            return

        # その他の対応拡張子は constants で定義されたローダーを使う
        loader_ctor = ct.SUPPORTED_EXTENSIONS.get(ext)
        if loader_ctor is None:
            print(f"⊘ ローダが未定義: {path}")
            return

        loader = loader_ctor(path)
        loaded = loader.load()
        if not loaded:
            print(f"⚠️ ローダは成功したがドキュメント0件: {path}")
        else:
            docs_all.extend(loaded)
            print(f"✓ ローダで読み込み完了: {path} (追加件数={len(loaded)})")

    except Exception as e:
        print(f"⚠️ file_load エラー: path={path} error={e}")


def adjust_string(s):
    """
    Windows環境でRAGが正常動作するよう調整
    
    Args:
        s: 調整を行う文字列
    
    Returns:
        調整を行った文字列
    """
    # 調整対象は文字列のみ
    if type(s) is not str:
        return s

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s
    
    # OSがWindows以外の場合はそのまま返す
    return s