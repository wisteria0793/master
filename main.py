import subprocess
import sys
import os

def main():
    """
    src/data_analysis/lda_topic_modeling.py を実行するためのメイン関数です。
    """
    # main.pyからの相対パスでスクリプトの場所を指定します
    script_path = os.path.join('src', 'data_analysis', 'lda_topic_modeling.py')

    # スクリプトが存在するか確認します
    if not os.path.exists(script_path):
        print(f"エラー: スクリプトが見つかりません: {script_path}")
        return

    print(f"スクリプトを実行します: {script_path}")

    try:
        # 現在のPythonインタプリタを使ってスクリプトを実行します
        # これにより、仮想環境が正しく使用されます
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,  # 標準出力と標準エラーをキャプチャします
            text=True,            # 出力をテキストとしてデコードします
            check=True,           # 実行が失敗した場合に例外を発生させます
            encoding='utf-8'      # 出力のエンコーディングを指定します
        )

        print("--- スクリプトの標準出力 ---")
        print(result.stdout)

        if result.stderr:
            print("--- スクリプトの標準エラー出力 ---")
            print(result.stderr)

        print("\nスクリプトは正常に終了しました。")

    except FileNotFoundError:
        print(f"エラー: Pythonインタプリタが見つかりません: '{sys.executable}'")
    except subprocess.CalledProcessError as e:
        print("\nスクリプトの実行中にエラーが発生しました。")
        print(f"終了コード: {e.returncode}")
        print("--- 標準出力 ---")
        print(e.stdout)
        print("--- 標準エラー出力 ---")
        print(e.stderr)
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")


if __name__ == "__main__":
    main()
