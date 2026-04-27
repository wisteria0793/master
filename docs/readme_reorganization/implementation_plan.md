# README.md の整理と実験アーカイブ化の実装計画

## 概要
現在、プロトタイプ開発に従事しており、`README.md` が肥大化しているため、内容を整理します。具体的には、実験1から実験15までの内容をアーカイブ用のドキュメント（`docs/history/experiments_archive.md`）に移動し、現在の `README.md` には研究概要と実験16以降の内容を記載するようにします。

## ユーザーレビューが必要な事項
- アーカイブ用ファイルの保存場所: `docs/history/experiments_archive.md`
- メインの `README.md` に残す内容: タイトル、研究テーマ、目的、および実験16以降。
- 追加する項目: 過去の実験へのリンク。

## 変更内容

### ドキュメント構成

#### [NEW] [experiments_archive.md](file:///Users/atsuyakatougi/Desktop/master/docs/history/experiments_archive.md)
`README.md` から切り出した実験1〜15の内容を保持します。

#### [MODIFY] [README.md](file:///Users/atsuyakatougi/Desktop/master/README.md)
実験1〜15を削除し、アーカイブへのリンクを追加します。実験16以降の内容はそのまま残します。

## 検証計画

### 目視確認
- `docs/history/experiments_archive.md` に実験1〜15が正しく抽出されているか確認。
- `README.md` の冒頭（タイトル・目的）が維持されているか確認。
- `README.md` からアーカイブへのリンクが機能するか確認。
- 実験16以降の内容が欠落なく表示されているか確認。
