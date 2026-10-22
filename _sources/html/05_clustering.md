# 文書分類

```{admonition} 今回のゴール
- 文書をラベルなしで自動的に分類する「クラスタリング」の考え方を理解する  
- TF-IDFなどのベクトル表現を用いた文書の類似度計算を復習する  
- K-meansクラスタリングおよびLDA（トピックモデル）の基本的な仕組みと実装を学ぶ  
```

## 文書分類とは
文書分類（document classification）とは、**複数の文章や記事を内容に応じてグループ化・分類する手法**。  
大量のテキストデータから意味のあるパターンを見つけることを目的としている。

### 文書分類の2つのアプローチ

| 種類 | 概要 | 代表的な手法 | 例 |
|------|------|---------------|----|
| **教師あり学習** | あらかじめラベル（カテゴリ）付きのデータで学習し、新しい文書を自動分類する。 | ロジスティック回帰、SVM、BERTなど | 「ニュース記事を政治・経済・スポーツに分類」 |
| **教師なし学習** | ラベルのない文書を自動的にグループ化し、内容の傾向を発見する。 | K-means、階層的クラスタリング、LDAなど | 「SNS投稿を話題ごとにまとめる」 |

今回の講義では、後者の **教師なし学習（unsupervised learning）** の手法を中心に扱う。  
特に、**K-meansクラスタリング** と **LDA（トピックモデル）** に焦点を当て、  
「似た内容の文書が自動的に集まる」仕組みを理解する。

### 文書分類の応用例
- ニュース記事やSNS投稿の自動グループ化  
- 顧客のレビューを話題ごとに整理  
- 論文・レポートのテーマ分析  
- 大規模なテキストデータからの知識発見

---

### 文書クラスタリングの考え方
クラスタリングでは、ラベルのない文書を「類似度」に基づいて自動的に分類する。  
目的は、文書群の中に存在する**潜在的なトピック構造**を発見すること。

#### 主な手法
- **K-meansクラスタリング**：文書をK個のグループに分ける（中心との距離で分類）  
- **階層的クラスタリング**：似た文書から順にまとめていく  
- **LDA（Latent Dirichlet Allocation）**：トピックモデルの代表例

---
## K-meansクラスタリング

### K-meansクラスタリングの流れ
1. 文書をTF-IDFベクトルに変換  
2. K個のクラスタ中心を初期化  
3. 各文書を最も近い中心に割り当て  
4. 各クラスタの平均ベクトルを再計算  
5. 割り当てが安定するまで繰り返す

---

### プログラム例（K-meansクラスタリング）

**ライブラリインストール（必要に応じて）**
```python
!pip -q install -U spacy ginza ja-ginza
```

**プログラム**  
```python
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd

# GiNZAの日本語モデルをロード
nlp = spacy.load("ja_ginza")

# 対象文書
texts = [
    "昨日の夜はラーメンを食べた。ラーメンは本当に美味しい。",
    "東京のラーメン店は人気がある。私はラーメンが大好きだ。",
    "野球の試合を見た。選手たちのプレーが素晴らしかった。",
    "サッカーの試合を観戦した。スタジアムの雰囲気が最高だった。",
]

# 形態素解析：名詞・固有名詞・形容詞・動詞を抽出
target_pos = ("NOUN", "PROPN", "ADJ", "VERB")
stop_words = ["\n", " ", "　"]

processed = []
for text in texts:
    doc = nlp(text)
    terms = []
    for token in doc:
        if token.pos_ in target_pos and token.lemma_ not in stop_words:
            terms.append(token.lemma_)
    processed.append(" ".join(terms))

# TF-IDFを計算
def split_by_space(text):
    return text.split()   # 空白で分割してリストにする

vectorizer = TfidfVectorizer(tokenizer=split_by_space, token_pattern=None, lowercase=False)
X = vectorizer.fit_transform(processed)

# TF-IDFをDataFrame化
df_tfidf = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
doc_ids = []
for i in range(len(texts)):
    doc_ids.append(f"D{i+1}")
df_tfidf.index = doc_ids

# TF-IDFの値
print("＜TF-IDF＞")
display(df_tfidf)

# K-meansクラスタリング
K=2
kmeans = KMeans(n_clusters=K, n_init=10, random_state=0)
labels = kmeans.fit_predict(X)

# 文書ごとのクラスタを表示（DataFrameで保存）
df_result = pd.DataFrame({
    "文書ID": doc_ids,
    "文書": texts,
    "クラスタ": labels
})

# クラスタリング結果
print("\n＜クラスタリング結果＞")
display(df_result)

# 補足：各クラスタの代表語を表示
terms = vectorizer.get_feature_names_out()
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

top_words = []
for i in range(K):
    top_terms = []
    for ind in order_centroids[i, :5]:
        top_terms.append(terms[ind])
    top_words.append(", ".join(top_terms))

df_top = pd.DataFrame({"クラスタ": range(K), "上位語": top_words})
print("\n＜各クラスタの上位語＞")
display(df_top)

```

**出力**  
＜TF-IDF＞
| |ある|たち|サッカー|スタジアム|プレー|ラーメン|人気|夜|大好き|店|昨日|最高|本当|東京|素晴らしい|美味しい|見る|観戦|試合|選手|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|D1|0\.0|0\.0|0\.0|0\.0|0\.0|0\.5762982154690183|0\.0|0\.3654806060100111|0\.0|0\.0|0\.3654806060100111|0\.0|0\.3654806060100111|0\.0|0\.0|0\.3654806060100111|0\.0|0\.0|0\.0|0\.0|
|D2|0\.3654806060100111|0\.0|0\.0|0\.0|0\.0|0\.5762982154690183|0\.3654806060100111|0\.0|0\.3654806060100111|0\.3654806060100111|0\.0|0\.0|0\.0|0\.3654806060100111|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|
|D3|0\.0|0\.388614292631317|0\.0|0\.0|0\.388614292631317|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.388614292631317|0\.0|0\.388614292631317|0\.0|0\.3063879719831814|0\.388614292631317|
|D4|0\.0|0\.0|0\.4217647821447532|0\.4217647821447532|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.4217647821447532|0\.0|0\.0|0\.0|0\.0|0\.0|0\.4217647821447532|0\.3325241986862672|0\.0|



＜クラスタリング結果＞
| 文書ID | 文書                              | クラスタ |
| :--: | :------------------------------ | :--: |
|  D1  | 昨日の夜はラーメンを食べた。ラーメンは本当に美味しい。     |   1  |
|  D2  | 東京のラーメン店は人気がある。私はラーメンが大好きだ。     |   1  |
|  D3  | 野球の試合を見た。選手たちのプレーが素晴らしかった。      |   0  |
|  D4  | サッカーの試合を観戦した。スタジアムの雰囲気が最高だった。 |   0  |


＜各クラスタの上位語＞
|  |                         |
| :--: | :-------------------------- |
|   0  | 試合, 観戦, 雰囲気, サッカー, スタジアム |
|   1  | ラーメン, 店, 食べる, 東京, 本当          |




**→ 「食べ物系」と「スポーツ系」に自然に分かれている。**

---

### 演習用データ（食・スポーツ・映画の18文書）で試す

先ほどの4文書は「ラーメン」「試合」のように話題がはっきり分かれる例だった。ここでは、演習4・5とLDAで共通に使う **18文書（食・スポーツ・映画 各6文書）** で試す。

**この例文のつくり方（クラスタリングがうまくいくポイント）**
- 同じ話題の文書が、**共通の単語**を何度も使っている（例：食→「料理・味・店・メニュー」、スポーツ→「試合・選手・チーム・応援」、映画→「映画・俳優・演技・ストーリー」）
- 話題をまたいで出てくる単語（「人気」「友人」「週末」など）を避けている
- 文書の数をある程度（18件）確保している

**データ**：[data-kmeans.csv](https://mayster8.github.io/nlp/data/data-kmeans.csv)（列：`id`, `text`）。下のプログラムは、このURLから直接読み込む。手元のファイルを使う場合は、Google Colab に配置して `pd.read_csv("data-kmeans.csv", encoding="utf-8-sig")` のようにファイル名を指定する。なお、このcsvは文字コードが UTF-8（BOM付き）なので、`encoding="utf-8-sig"` を指定している（Excelで開いても文字化けしない）。

**プログラム**
```python
import spacy, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

nlp = spacy.load("ja_ginza")

# CSVから文書を読み込む（URLから直接読み込める）
url = "https://mayster8.github.io/nlp/data/data-kmeans.csv"
df = pd.read_csv(url, encoding="utf-8-sig")
texts = df["text"].to_list()

# 形態素解析：名詞・固有名詞・形容詞・動詞を抽出
target_pos = ("NOUN", "PROPN", "ADJ", "VERB")
stop_words = ["\n", " ", "　"]

processed = []
for text in texts:
    doc = nlp(text)
    terms = []
    for token in doc:
        if token.pos_ in target_pos and token.lemma_ not in stop_words:
            terms.append(token.lemma_)
    processed.append(" ".join(terms))

# TF-IDF
def split_by_space(text):
    return text.split()   # 空白で分割してリストにする

# min_df=2：1つの文書にしか出ない語は除く
vectorizer = TfidfVectorizer(tokenizer=split_by_space, token_pattern=None, lowercase=False, min_df=2)
X = vectorizer.fit_transform(processed)

# K-means（初期値による結果のぶれを減らすため n_init=10）
K = 3
kmeans = KMeans(n_clusters=K, n_init=10, random_state=0)
labels = kmeans.fit_predict(X)

doc_ids = []
for i in range(len(texts)):
    doc_ids.append(f"D{i+1}")
df_result = pd.DataFrame({"文書ID": doc_ids,
                          "文書": texts, "クラスタ": labels})
print("＜クラスタリング結果＞")
display(df_result.sort_values("クラスタ").reset_index(drop=True))

# 各クラスタの上位語
terms = vectorizer.get_feature_names_out()
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
top_words = []
for i in range(K):
    top_terms = []
    for ind in order_centroids[i, :5]:
        top_terms.append(terms[ind])
    top_words.append(", ".join(top_terms))

df_top = pd.DataFrame({"クラスタ": range(K), "上位語": top_words})
print("\n＜各クラスタの上位語＞")
display(df_top)
```

**出力**（クラスタ番号は実行ごとに入れ替わることがある）

＜クラスタリング結果＞

| 文書ID | 文書 | クラスタ |
| :--: | :-- | :--: |
| D1 | ドラマの最終回を観ました。俳優の演技とストーリーに感動し、主人公の姿に涙が出ました。 | 1 |
| D2 | 駅前のレストランで食事をしました。メニューの種類が多く、料理の味も満足でした。 | 2 |
| D3 | このカフェはランチのメニューが豊富です。料理もデザートも味が良く、店の雰囲気も素敵です。 | 2 |
| D4 | 話題の映画を観ました。ストーリーが面白く、監督の演出が印象的な作品でした。 | 1 |
| D5 | 定食屋のランチは、味も量も大満足です。毎日メニューが変わり、料理の種類も豊富です。 | 2 |
| D6 | ラグビーの試合は迫力満点でした。両チームの選手がぶつかり合い、最後の得点で勝敗が決まりました。 | 0 |
| D7 | 新作の映画を映画館で観ました。映像が美しく、俳優の演技も素晴らしい作品でした。 | 1 |
| D8 | イタリアンの店でパスタを食べました。ソースの味とデザートの種類に満足し、また食事に来たいです。 | 2 |
| D9 | 駅前のラーメン店でランチを食べました。スープの味が濃厚で、料理の量もたっぷりでした。 | 2 |
| D10 | サッカーの試合をスタジアムで観戦しました。チームが勝利し、選手の得点に応援も盛り上がりました。 | 0 |
| D11 | テニスの大会で選手が優勝しました。試合の応援に行き、勝利の瞬間を見ることができました。 | 0 |
| D12 | 今日の試合は接戦でした。応援しているチームの選手が最後に得点して勝利しました。 | 0 |
| D13 | 配信サービスで話題の作品を観ました。俳優の演技が見事で、映画のようなストーリーでした。 | 1 |
| D14 | 寿司店で夕食を楽しみました。新鮮な料理と丁寧な味付けで、店の評判にも納得しました。 | 2 |
| D15 | アニメ映画を観ました。映像の迫力と主題歌の音楽が素晴らしく、ストーリーも感動的な作品でした。 | 1 |
| D16 | この作品は映像と音楽が魅力の映画です。主人公のストーリーに引き込まれ、俳優の演技にも感動しました。 | 1 |
| D17 | 野球の試合をテレビで応援しました。選手の活躍でチームが勝利し、最高の試合でした。 | 0 |
| D18 | バスケットボールの試合で選手が大活躍しました。チームの勝利に観客の応援も大きくなりました。 | 0 |

＜各クラスタの上位語＞

| クラスタ | 上位語 |
| :--: | :-- |
| 0 | 試合, 選手, 勝利, 応援, チーム |
| 1 | 観る, ストーリー, 作品, 映画, 演技 |
| 2 | 料理, 店, 味, メニュー, ランチ |

**→ 「スポーツ」「映画」「食」の3つに、すべての文書が正しく分かれている。**

> **補足：`min_df=2` について**  
> `min_df=2` は「2つ以上の文書に出てくる語だけを使う」という指定。1つの文書にしか出ない語は、他の文書との類似度の手がかりにならないので除く。この18文書では結果はほとんど変わらないが、アンケートの回答のような短い文では、語が散らばってクラスタが分かれにくくなるため、効果が大きくなる。

> **補足：`n_init=10` について**  
> K-means は最初のクラスタ中心を乱数で決めるため、`n_init=1` だと実行のたびに結果が変わることがある。`n_init=10` とすると、初期値を10回変えて最も良い結果を採用するので、結果が安定する。

---

## 階層的クラスタリング

### 階層的クラスタリングとは
文書同士の「似ている関係」を順にまとめていくことで、**文書の関係構造をツリー（木）として表す**手法  
K-means のように「クラスタ数K」を先に決めなくても、後から**どの高さで区切るか**を選べるのが特徴  

---

### クラスタリングの流れ
1. 最初は、すべての文書を1つずつ別のグループにする。  
2. 最も似ている2つのグループを1つにまとめる。  
3. それを繰り返していくと、「どの文書がどの順でまとめられたか」がわかる木（デンドログラム）ができる。  
4. 木の高さで区切ることで、クラスタ数を変えることができる。

---

### 代表的な方法
階層的クラスタリングには2つの方法があるが、ここでは一般的な**凝集型**を扱う。

- **凝集型**：小さいグループを少しずつ結合していく（下から上に作る）
- **分割型**：全体から順に分けていく（上から下）

---

### 文書の類似度の測り方
文書を数値ベクトル（TF-IDF）に変換して、**コサイン距離（1 − コサイン類似度）**を使って比較する。  
これは「使っている単語の方向がどれくらい似ているか」を表すもので、文章の長さにあまり影響を受けない。

---

### プログラム例（階層的クラスタリング）

**ライブラリのインストール（図における日本語文字化け対応）**  
```python
!pip -q install japanize-matplotlib
```

**プログラム**  
```python
import spacy
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import pandas as pd

nlp = spacy.load("ja_ginza")

# 対象文書
texts = [
    "昨日の夜はラーメンを食べた。ラーメンは本当に美味しい。",
    "東京のラーメン店は人気がある。私はラーメンが大好きだ。",
    "野球の試合を見た。選手たちのプレーが素晴らしかった。",
    "サッカーの試合を観戦した。スタジアムの雰囲気が最高だった。",
]

# 形態素解析：名詞・固有名詞・形容詞・動詞を抽出
target_pos = ("NOUN", "PROPN", "ADJ", "VERB")
stop_words = ["\n", " ", "　"]

processed = []
for text in texts:
    doc = nlp(text)
    terms = []
    for token in doc:
        if token.pos_ in target_pos and token.lemma_ not in stop_words:
            terms.append(token.lemma_)
    processed.append(" ".join(terms))

# TF-IDFベクトルを作成
def split_by_space(text):
    return text.split()   # 空白で分割してリストにする

vectorizer = TfidfVectorizer(tokenizer=split_by_space, token_pattern=None, lowercase=False)
X = vectorizer.fit_transform(processed)

# コサイン距離を計算して階層的クラスタリング
D = cosine_distances(X)
Z = linkage(squareform(D, checks=False), method="average")

# デンドログラムを描画
plt.figure(figsize=(8, 4))
doc_ids = []
for i in range(len(texts)):
    doc_ids.append(f"D{i+1}")
dendrogram(Z, labels=doc_ids)
plt.title("階層的クラスタリング")
plt.show()

# == クラスタリング結果の抽出 ==
# クラスタ数で指定したい場合
labels = fcluster(Z, t=2, criterion="maxclust")
# 距離のしきい値で指定したい場合
# fcluster(Z, t=0.5, criterion="distance")

# 結果をDataFrameで表示
df_result = pd.DataFrame({
    "文書ID": doc_ids,
    "文書": texts,
    "クラスタ": labels
})

print("＜クラスタリング結果＞")
print(df_result)
```

**出力**

<img src=../static/out-clustering.png>

＜クラスタリング結果＞  
| 文書ID | 文書                              | クラスタ |
| :--: | :------------------------------ | :--: |
|  D1  | 昨日の夜はラーメンを食べた。ラーメンは本当に美味しい。     |   1  |
|  D2  | 東京のラーメン店は人気がある。私はラーメンが大好きだ。     |   1  |
|  D3  | 野球の試合を見た。選手たちのプレーが素晴らしかった。      |   2  |
|  D4  | サッカーの試合を観戦した。スタジアムの雰囲気が最高だった。 |   2  |
---

### デンドログラムの読み方
- 葉（下のラベル）＝各文書  
- 枝の高さ＝その文書どうし（またはクラスタ）がどれくらい似ているか（距離）  
- 高さの低い部分で結合しているほど似ている  
- **切る高さ**を変えることでクラスタ数を調整できる  

---

### 18文書で試す（演習5と同じデータ）

K-meansと同じ [data-kmeans.csv](https://mayster8.github.io/nlp/data/data-kmeans.csv)（18文書）を使い、階層的クラスタリングで樹形図を作る。

**プログラム**
```python
import spacy, pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

nlp = spacy.load("ja_ginza")
# CSVから文書を読み込む（URLから直接読み込める）
url = "https://mayster8.github.io/nlp/data/data-kmeans.csv"
df = pd.read_csv(url, encoding="utf-8-sig")
texts = df["text"].to_list()

target_pos = ("NOUN", "PROPN", "ADJ", "VERB")
stop_words = ["\n", " ", "　"]
processed = []
for text in texts:
    doc = nlp(text)
    terms = []
    for t in doc:
        if t.pos_ in target_pos and t.lemma_ not in stop_words:
            terms.append(t.lemma_)
    processed.append(" ".join(terms))

def split_by_space(text):
    return text.split()   # 空白で分割してリストにする

# min_df=2：2つ以上の文書に出てくる語だけを使う（1つの文書にしか出ない語は除く）
vectorizer = TfidfVectorizer(tokenizer=split_by_space, token_pattern=None, lowercase=False, min_df=2)
X = vectorizer.fit_transform(processed)

# コサイン距離で階層的クラスタリング
D = cosine_distances(X)
Z = linkage(squareform(D, checks=False), method="average")

plt.figure(figsize=(10, 4))
doc_ids = []
for i in range(len(texts)):
    doc_ids.append(f"D{i+1}")
dendrogram(Z, labels=doc_ids)
plt.title("階層的クラスタリング")
plt.show()

labels = fcluster(Z, t=3, criterion="maxclust")
df_result = pd.DataFrame({"文書ID": doc_ids,
                          "文書": texts, "クラスタ": labels})
print("＜クラスタリング結果＞")
print(df_result.sort_values("クラスタ").to_string(index=False))
```

**出力**

<img src=../static/out-clustering-18.png>

＜クラスタリング結果（3クラスタ）＞

| 文書ID | 文書 | クラスタ |
| :--: | :-- | :--: |
| D1 | ドラマの最終回を観ました。俳優の演技とストーリーに感動し、主人公の姿に涙が出ました。 | 3 |
| D2 | 駅前のレストランで食事をしました。メニューの種類が多く、料理の味も満足でした。 | 1 |
| D3 | このカフェはランチのメニューが豊富です。料理もデザートも味が良く、店の雰囲気も素敵です。 | 1 |
| D4 | 話題の映画を観ました。ストーリーが面白く、監督の演出が印象的な作品でした。 | 3 |
| D5 | 定食屋のランチは、味も量も大満足です。毎日メニューが変わり、料理の種類も豊富です。 | 1 |
| D6 | ラグビーの試合は迫力満点でした。両チームの選手がぶつかり合い、最後の得点で勝敗が決まりました。 | 2 |
| D7 | 新作の映画を映画館で観ました。映像が美しく、俳優の演技も素晴らしい作品でした。 | 3 |
| D8 | イタリアンの店でパスタを食べました。ソースの味とデザートの種類に満足し、また食事に来たいです。 | 1 |
| D9 | 駅前のラーメン店でランチを食べました。スープの味が濃厚で、料理の量もたっぷりでした。 | 1 |
| D10 | サッカーの試合をスタジアムで観戦しました。チームが勝利し、選手の得点に応援も盛り上がりました。 | 2 |
| D11 | テニスの大会で選手が優勝しました。試合の応援に行き、勝利の瞬間を見ることができました。 | 2 |
| D12 | 今日の試合は接戦でした。応援しているチームの選手が最後に得点して勝利しました。 | 2 |
| D13 | 配信サービスで話題の作品を観ました。俳優の演技が見事で、映画のようなストーリーでした。 | 3 |
| D14 | 寿司店で夕食を楽しみました。新鮮な料理と丁寧な味付けで、店の評判にも納得しました。 | 1 |
| D15 | アニメ映画を観ました。映像の迫力と主題歌の音楽が素晴らしく、ストーリーも感動的な作品でした。 | 3 |
| D16 | この作品は映像と音楽が魅力の映画です。主人公のストーリーに引き込まれ、俳優の演技にも感動しました。 | 3 |
| D17 | 野球の試合をテレビで応援しました。選手の活躍でチームが勝利し、最高の試合でした。 | 2 |
| D18 | バスケットボールの試合で選手が大活躍しました。チームの勝利に観客の応援も大きくなりました。 | 2 |

**→ 樹形図でも、「食」「スポーツ」「映画」の3つのまとまりが大きな枝として分かれている。**

---

### K-meansとの違い

| 比較項目 | K-means | 階層的クラスタリング |
|-----------|----------|--------------------|
| クラスタ数 | あらかじめ指定する | デンドログラムを見て後から決める |
| 結果の形 | 中心（重心）で分ける | 木構造で関係を表現 |
| 向いている規模 | 大量データ | 文書数が少ない場合（〜数百件） |
| 可視化 | クラスタ番号のみ | デンドログラムで関係がわかる |

---

---

## トピックモデル（LDA：Latent Dirichlet Allocation）

### トピックモデルとは？

クラスタリングでは「どの文書がどのクラスタに属するか」を求めるが、  
**LDA（潜在的ディリクレ配分法）** では「各文書がどのトピックをどの割合で含むか」を推定する。

#### 特徴
- 各文書は複数のトピックを**混合している**と仮定。  
- 各トピックは特定の単語分布によって表される。  
- トピック構造を確率的に推定する。

#### イメージ
| 文書 | トピック1（料理） | トピック2（スポーツ） |
|------|------------------|--------------------|
| D1: 「お寿司を食べた」 | 0.9 | 0.1 |
| D2: 「野球の試合を見た」 | 0.2 | 0.8 |

---

### プログラム例（LDAによるトピックモデル）

K-means・階層的クラスタリングと同じ [data-kmeans.csv](https://mayster8.github.io/nlp/data/data-kmeans.csv)（18文書）を使う。LDAは文書数・語数が少ないと結果が不安定になりやすいので、**1つの文書にしか出ない単語を除く**（`min_df=2`）など、いくつか設定を工夫している。

**プログラム**  

```python
import spacy, pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

nlp = spacy.load("ja_ginza")
# CSVから文書を読み込む（URLから直接読み込める）
url = "https://mayster8.github.io/nlp/data/data-kmeans.csv"
df = pd.read_csv(url, encoding="utf-8-sig")
texts = df["text"].to_list()

# 形態素解析：名詞・固有名詞を抽出（原形）
target_pos = ("NOUN", "PROPN")
stop_words = ["\n", " ", "　"]

def tokenize(text):
    doc = nlp(text)
    terms = []
    for t in doc:
        if t.pos_ in target_pos and t.lemma_ not in stop_words:
            terms.append(t.lemma_)
    return " ".join(terms)

processed = []
for t in texts:
    processed.append(tokenize(t))

# Bag of Words（min_df=2：1つの文書にしか出ない語は除く）
def split_by_space(text):
    return text.split()   # 空白で分割してリストにする

vectorizer = CountVectorizer(tokenizer=split_by_space, token_pattern=None,
                             lowercase=False, min_df=2)
X = vectorizer.fit_transform(processed)

# LDAモデル
K = 3
lda = LatentDirichletAllocation(n_components=K, max_iter=200, learning_method="batch",
                                doc_topic_prior=0.1, topic_word_prior=1.0, random_state=0)
lda.fit(X)

# 文書ごとのトピック分布（DataFrameで表示）
topic_cols = []
for i in range(K):
    topic_cols.append(f"トピック{i}")
df_topics = pd.DataFrame(lda.transform(X), columns=topic_cols).round(2)
doc_ids = []
for i in range(len(texts)):
    doc_ids.append(f"D{i+1}")
df_topics.insert(0, "文書ID", doc_ids)
doc_heads = []
for t in texts:
    doc_heads.append(t[:18] + "…")
df_topics["文書"] = doc_heads
print("＜文書ごとのトピック分布＞")
display(df_topics)

# 各トピックの上位語（DataFrameで表示）
words = vectorizer.get_feature_names_out()
top_words = []
for topic in lda.components_:
    top = []
    for j in topic.argsort()[::-1][:6]:
        top.append(words[j])
    top_words.append(", ".join(top))
df_words = pd.DataFrame({
    "トピック": range(K),
    "上位語": top_words,
})
print("＜各トピックの上位語＞")
display(df_words)

# 発展：2つの話題が混ざった新しい文書のトピック分布（DataFrameで表示）
new_text = "試合を観戦したあと、チームの選手と一緒に店で食事をしました。"
x_new = vectorizer.transform([tokenize(new_text)])
df_new = pd.DataFrame(lda.transform(x_new).round(2), columns=topic_cols)
df_new.insert(0, "文書", new_text)
print("＜新しい文書のトピック分布＞")
display(df_new)
```

> **補足：LDAを安定させる設定**
> - `min_df=2`：1つの文書にしか出てこない単語は、トピックの手がかりにならないので除く
> - `doc_topic_prior=0.1`：「1つの文書は少数のトピックに偏る」ことを仮定する（小さいほど偏る）
> - `topic_word_prior=1.0`：「各トピックはある程度いろいろな単語を使う」ことを仮定する
> - `max_iter=200`, `learning_method="batch"`：繰り返し回数を十分にとり、結果を安定させる
> - `random_state=0`：乱数を固定する（LDAは初期値によって結果が変わることがある）

---

**出力**  

＜文書ごとのトピック分布＞

| 文書ID | トピック0 | トピック1 | トピック2 | 文書（先頭） |
|:--:|:--:|:--:|:--:|:--|
| D1 | 0.95 | 0.02 | 0.02 | ドラマの最終回を観ました。俳優の演技… |
| D2 | 0.02 | 0.02 | 0.97 | 駅前のレストランで食事をしました。メ… |
| D3 | 0.02 | 0.02 | 0.97 | このカフェはランチのメニューが豊富で… |
| D4 | 0.95 | 0.02 | 0.02 | 話題の映画を観ました。ストーリーが面… |
| D5 | 0.01 | 0.01 | 0.97 | 定食屋のランチは、味も量も大満足です… |
| D6 | 0.02 | 0.97 | 0.02 | ラグビーの試合は迫力満点でした。両チ… |
| D7 | 0.96 | 0.02 | 0.02 | 新作の映画を映画館で観ました。映像が… |
| D8 | 0.02 | 0.02 | 0.96 | イタリアンの店でパスタを食べました。… |
| D9 | 0.02 | 0.02 | 0.97 | 駅前のラーメン店でランチを食べました… |
| D10 | 0.02 | 0.96 | 0.02 | サッカーの試合をスタジアムで観戦しま… |
| D11 | 0.02 | 0.95 | 0.02 | テニスの大会で選手が優勝しました。試… |
| D12 | 0.02 | 0.95 | 0.02 | 今日の試合は接戦でした。応援している… |
| D13 | 0.97 | 0.02 | 0.02 | 配信サービスで話題の作品を観ました。… |
| D14 | 0.04 | 0.04 | 0.91 | 寿司店で夕食を楽しみました。新鮮な料… |
| D15 | 0.97 | 0.02 | 0.02 | アニメ映画を観ました。映像の迫力と主… |
| D16 | 0.98 | 0.01 | 0.01 | この作品は映像と音楽が魅力の映画です… |
| D17 | 0.02 | 0.95 | 0.02 | 野球の試合をテレビで応援しました。選… |
| D18 | 0.02 | 0.97 | 0.02 | バスケットボールの試合で選手が大活躍… |

＜各トピックの上位語＞

| トピック | 上位語 |
| :--: | :-- |
| 0 | 映画, 作品, ストーリー, 演技, 俳優, 映像 |
| 1 | 試合, 選手, チーム, 応援, 得点, 最後 |
| 2 | 味, 料理, 店, メニュー, ランチ, 種類 |

**→ トピック0が「映画」、トピック1が「スポーツ」、トピック2が「食」と解釈できる。**  
（トピックの番号は実行ごとに入れ替わることがある。上位語を見て意味を解釈するのは人間の役割）

＜新しい文書のトピック分布（発展）＞

学習済みのモデルに、**2つの話題が混ざった文書**を入力してみる。

> 「試合を観戦したあと、チームの選手と一緒に店で食事をしました。」

| 文書 | トピック0 | トピック1 | トピック2 |
|:--|:--:|:--:|:--:|
| 試合を観戦したあと、チームの選手と一緒に店で食事をしました。 | 0.02 | 0.72 | 0.27 |

→ K-meansでは「スポーツ」か「食」のどちらか1つに割り当てられるが、LDAでは「スポーツ約7割・食約3割」のように、**複数のトピックの混合**として表せる。

---

## クラスタリングとトピックモデルの違い

| 観点 | K-means（クラスタリング） | LDA（トピックモデル） |
|------|----------------------------|----------------------|
| 文書の所属 | 1つのクラスタに属する | 複数トピックを混合 |
| 手法の性質 | 幾何的（距離ベース） | 確率的（分布ベース） |
| 出力 | クラスタごとの文書集合 | トピックごとの単語分布 |
| 解釈 | グループ分類 | 話題の構造推定 |

---
