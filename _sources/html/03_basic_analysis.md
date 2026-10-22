# 前処理・形態素解析・特徴抽出

```{admonition} 今回のゴール
- テキスト前処理の必要性を理解する
- 形態素解析（spaCy + GiNZA）を体験する
- Bag-of-Words（BoW）と TF-IDF の仕組みを理解する
```

---

## テキスト前処理の重要性

### なぜ前処理が必要か？
- 生のテキストはそのままでは分析しにくい。
- 句読点・記号・助詞などのノイズが多い。
- 表記ゆれ（例：スマホ／スマートフォン）が存在する。

### 主な前処理の例
- 正規化（全角・半角の統一、かな種の統一 など）
- ストップワード除去（「する」「ある」「こと」など意味が薄い語）
- 表記ゆれ処理（「出来る」⇔「できる」、「東京都市大学」⇔「都市大」）

---

## 形態素解析（GiNZA）

**ライブラリインストール（必要に応じて）**
```python
!pip -q install spacy ginza ja-ginza
```

### プログラム例（形態素の抽出）
```python
import spacy
nlp = spacy.load("ja_ginza")

doc = nlp("私は昨日、東京で美味しいラーメンを食べた。")

# 形態素（原形）を抽出してリストに追加
tokens = []
for token in doc:
    tokens.append(token.lemma_)

print(tokens)
```

**出力**
```text
['私', 'は', '昨日', '、', '東京', 'で', '美味しい', 'ラーメン', 'を', '食べる', 'た', '。']
```

---


### プログラム例（形態素の詳細表示）
```python
import spacy

nlp = spacy.load("ja_ginza")
doc = nlp("私は昨日、東京で美味しいラーメンを食べた。")

for token in doc:
    print(f"表層形: {token.text}\t原形: {token.lemma_}\tUPOS: {token.pos_}\t詳細品詞: {token.tag_}")
```

**出力**
```
表層形: 私      原形: 私      UPOS: PRON   詳細品詞: 名詞-代名詞-一般
表層形: 昨日    原形: 昨日    UPOS: ADV    詳細品詞: 名詞-副詞的-一般
表層形: 東京    原形: 東京    UPOS: PROPN  詳細品詞: 名詞-固有名詞-地名-一般
表層形: ラーメン 原形: ラーメン UPOS: NOUN  詳細品詞: 名詞-普通名詞-一般
表層形: 食べ    原形: 食べる  UPOS: VERB   詳細品詞: 動詞-一般
```

**補足**：f文字列を使った書き方

Python 3.6 以降で使える **フォーマット済み文字列** の書き方。
文字列の前に **`f`** を付け、`{}` の中に変数や式を直接書くことができる。

＜例＞
```python
name = "田中"
age = 20
print(f"名前は {name}、年齢は {age} 歳です。")
```

---


### プログラム例（改行を含む長文）
```python
import spacy
nlp = spacy.load("ja_ginza")

text = """
自然言語処理の基本概念や主要な分析手法について理解して、説明できる。
Pythonを用いて自然言語処理の基本的なプログラムを実装して、実データを分析できる。
自然言語処理の技術が応用される分野や課題について考察して、将来の応用可能性を議論できる。
"""

doc = nlp(text)

# 形態素（原形）を抽出してリストに追加
tokens = []
for token in doc:
    tokens.append(token.lemma_)

print(tokens)
```

**出力**
```text
['\n', '自然言語処理', 'の', '基本', '概念', 'や', '主要', 'だ', '分析', '手法', 'に', 'つく', 'て', '理解', 'する', 'て', '、', '説明', 'できる', '。', '\n', 'python', 'を', '用いる', 'て', '自然言語処理', 'の', '基本的', 'だ', 'プログラム', 'を', '実装', 'する', 'て', '、', '実', 'データ', 'を', '分析', 'できる', '。', '\n', '自然言語処理', 'の', '技術', 'が', '応用', 'する', 'れる', '分野', 'や', '課題', 'に', 'つく', 'て', '考察', 'する', 'て', '、', '将来', 'の', '応用', '可能性', 'を', '議論', 'できる', '。', '\n']
```

上記のプログラムだと、テキストに改行文字`\n`が含まれてしまう。
そのため、品詞の種類やストップワードで抽出する用語を選定する。

### プログラム例（特定の品詞、ストップワード対応）

＜例＞
```python
import spacy
nlp = spacy.load("ja_ginza")

text = """
自然言語処理の基本概念や主要な分析手法について理解して、説明できる。
Pythonを用いて自然言語処理の基本的なプログラムを実装して、実データを分析できる。
自然言語処理の技術が応用される分野や課題について考察して、将来の応用可能性を議論できる。
"""

stop_words = ["\n", "。", "、"]

doc = nlp(text)

# 形態素（原形）を抽出してリストに追加
tokens = []
for token in doc:

    # ストップワードに含まれていなければ追加
    if token.lemma_ not in stop_words:
        tokens.append(token.lemma_)

print(tokens)
```

**出力**
```text
['自然言語処理', 'の', '基本', '概念', 'や', '主要', 'だ', '分析', '手法', 'に', 'つく', 'て', '理解', 'する', 'て', '説明', 'できる', 'python', 'を', '用いる', 'て', '自然言語処理', 'の', '基本的', 'だ', 'プログラム', 'を', '実装', 'する', 'て', '実', 'データ', 'を', '分析', 'できる', '自然言語処理', 'の', '技術', 'が', '応用', 'する', 'れる', '分野', 'や', '課題', 'に', 'つく', 'て', '考察', 'する', 'て', '将来', 'の', '応用', '可能性', 'を', '議論', 'できる']
```


### プログラム例（複数文、特定の品詞、ストップワード対応）
```python
import spacy

nlp = spacy.load("ja_ginza")

# 解析対象の文書リスト
texts = [
    "昨日は台風が東京を直撃しました",
    "東京のラーメン 店が新商品を発表"
]

# 対象の品詞（名詞、固有名詞、形容詞）
target_pos = ("NOUN", "PROPN", "ADJ")

# ストップワード
stop_words = ["\n", " ", "　"]

# 各文書を処理
for text in texts:
    doc = nlp(text)
    terms = []
    
    for token in doc:

        # 対象品詞かつ、ストップワードは除外
        if (token.pos_ in target_pos) and (token.lemma_ not in stop_words):
            terms.append(token.lemma_)

    print(terms)
```

**出力**
```
['昨日', '台風', '東京']
['東京', 'ラーメン', '店', '新商品', '発表']
```

---

## 特徴抽出（BoW と TF-IDF）

テキストはそのままでは扱えないため、数値ベクトルに変換する。  
代表的な方法は **Bag-of-Words（BoW）** と **TF-IDF**。

### Bag-of-Words（BoW）
- 単語の出現回数を数える。
- 単語の順序は無視される。  

#### プログラム例（用語の頻度を計算）

```python
import spacy
from collections import Counter

# モデル読み込み
nlp = spacy.load("ja_ginza")

# 対象文書
texts = [
    "昨日の夜はラーメンを食べた。ラーメンは本当に美味しい。",
    "東京のラーメン店は人気がある。私はラーメンが大好きだ。"
]

# 対象の品詞（名詞、固有名詞、形容詞）
target_pos = ("NOUN", "PROPN", "ADJ")

# ストップワード
stop_words = ["\n", " ", "　"]

# 各文書を処理
doc_index = 0
for text in texts:
    doc_index += 1
    doc = nlp(text)

    # 単語リストを作成
    terms = []
    for token in doc:

        # 対象の品詞かつ、ストップワードは除外
        if (token.pos_ in target_pos) and (token.lemma_ not in stop_words):
            terms.append(token.lemma_)

    # 出現回数をカウント
    tf = Counter(terms)

    print(tf)
```

**出力**
```text
Counter({'ラーメン': 1, '美味しい': 1})
Counter({'ラーメン': 1, '高い': 1})
```

##### 補足：Counter の使い方

Python の標準ライブラリ **collections** に含まれる `Counter` は、  
リストやタプルの要素の出現回数を簡単に数えることができるクラス。

---

＜基本の使い方＞
```python
from collections import Counter

words = ["ラーメン", "美味しい", "ラーメン"]
tf = Counter(words)
print(tf)
```

出力
```
Counter({'ラーメン': 2, '美味しい': 1})
```


---

＜項目ごとのアクセス＞
```python
print(tf["ラーメン"])
print(tf["美味しい"])
```

出力
```
2
1
```

---

＜for文で回す＞
```python
for term, count in tf.items():
    print(term, count)
```

出力
```
ラーメン 2
美味しい 1
```

---

- `Counter` を使うと **単語の出現回数（Term Frequency, TF）** を簡単に数えられる  
- Bag-of-Words の実装でよく使われる  

---

#### プログラム例（用語の頻度を計算し、Data Frame に保存）

```python
import spacy
from collections import Counter
import pandas as pd

nlp = spacy.load("ja_ginza")

# 対象文書
texts = [
    "昨日の夜はラーメンを食べた。ラーメンは本当に美味しい。",
    "東京のラーメン店は人気がある。私はラーメンが大好きだ。"
]

# 対象の品詞（名詞、固有名詞、形容詞）
target_pos = ("NOUN", "PROPN", "ADJ")

# ストップワード
stop_words = ["\n", " ", "　"]

# 文書ごとの頻度辞書を保存
doc_counters = []

doc_index = 0
for text in texts:
    doc_index += 1
    doc = nlp(text)

    # 単語リストを作成
    terms = []
    for token in doc:
        if (token.pos_ in target_pos) and (token.lemma_ not in stop_words):
            terms.append(token.lemma_)

    # 出現回数をカウント
    tf = Counter(terms)
    doc_counters.append(tf)

# DataFrame に変換（行＝文書、列＝単語、セル＝頻度）
df = pd.DataFrame(doc_counters)
df = df.fillna(0).astype(int)   # 出現しない単語は 0 に

df
```

**出力**  
|    |   昨日 |   夜 |   ラーメン |   本当 |   美味しい |   東京 |   店 |   人気 |   大好き |
|---:|-------:|-----:|-----------:|-------:|-----------:|-------:|-----:|-------:|---------:|
|  0 |      1 |    1 |          2 |      1 |          1 |      0 |    0 |      0 |        0 |
|  1 |      0 |    0 |          2 |      0 |          0 |      1 |    1 |      1 |        1 |

---


### TF-IDF
- **TF（Term Frequency）**：文書内での単語の出現頻度
- **IDF（Inverse Document Frequency）**：多くの文書に現れる一般的な単語の重みを下げる指標
- **TF-IDF**：TF × IDF により、その文書に特徴的な単語を強調できる手法

＜例＞
```text
"昨日の夜はラーメンを食べた。ラーメンは本当に美味しい。"
"東京のラーメン店は人気がある。私はラーメンが大好きだ。"
```

例えば「ラーメン」という単語が複数の文書に出てきた場合、単純な TF では重要に見えるが、  
TF-IDF では全体に広く出ている単語の重みを下げ、より特徴的な単語（例：美味しい、大好き など）が強調される


#### 計算式

TF-IDF = TF × IDF

- **TF**（文書内頻度）  
  \\( TF(t, d) = \\frac{単語 t の出現回数}{文書 d 内の総単語数} \\)

- **IDF**（逆文書頻度）  
  \\( IDF(t) = \\log\\frac{総文書数}{単語 t を含む文書数 + 1} \\)

つまり：  
- よく出てくる単語 → TFが大きい  
- どの文書にも出てくる単語 → IDFが小さい  
- 特定の文書にだけ多く出る単語 → TF-IDFが大きい（＝特徴語）

---


#### プログラム例（TF-IDF の計算）


```python
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

nlp = spacy.load("ja_ginza")

# 対象文書
texts = [
    "昨日の夜はラーメンを食べた。ラーメンは本当に美味しい。",
    "東京のラーメン店は人気がある。私はラーメンが大好きだ。"
]

# 前処理：形態素解析で名詞・固有名詞・形容詞を抽出（原形）
target_pos = ("NOUN", "PROPN", "ADJ")
stop_words = ["\n", " ", "　"]

processed = []
for text in texts:
    doc = nlp(text)
    terms = []
    for token in doc:
        if token.pos_ in target_pos and token.lemma_ not in stop_words:
            terms.append(token.lemma_)
    processed.append(" ".join(terms))

# TF-IDF を計算
def split_by_space(text):
    return text.split()   # 空白で分割してリストにする

vectorizer = TfidfVectorizer(tokenizer=split_by_space, token_pattern=None, lowercase=False)
X = vectorizer.fit_transform(processed)

# DataFrameに変換
df_tfidf = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
df_tfidf
```

**出力**

|    |   ラーメン |     人気 |       夜 |   大好き |       店 |     昨日 |     本当 |     東京 |   美味しい |
|---:|-----------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|-----------:|
|  0 |   0.579739 | 0        | 0.407401 | 0        | 0        | 0.407401 | 0.407401 | 0        |   0.407401 |
|  1 |   0.579739 | 0.407401 | 0        | 0.407401 | 0.407401 | 0        | 0        | 0.407401 |   0        |


##### TfidfVectorizer

`TfidfVectorizer` はテキストデータを **TF-IDF ベクトル** に変換するためのクラスで、
`scikit-learn`（`sklearn.feature_extraction.text`）に含まれている。

機械学習モデルは文字列を直接扱えないため、
文章を「単語の重み」で数値化して特徴ベクトル化するために使う。


**日本語テキストでの注意点**

`TfidfVectorizer` はもともと英語向けに設計されており、
デフォルトでは次のような正規表現で単語を抽出する。

```
token_pattern = r"(?u)\b\w\w+\b"
```

この設定では、**2文字以上の単語のみ** を対象にするため、
日本語の1文字の単語（例：「夜」「私」など）は除外されてしまう。

---

GiNZAなどで形態素解析を行い、空白区切りで単語を並べた文字列を入力したい場合は、以下のように
`TfidfVectorizer` の内部トークン化を無効化する。

```python
def split_by_space(text):
    return text.split()   # 空白で分割してリストにする

vectorizer = TfidfVectorizer(
    tokenizer=split_by_space,          # 空白で分割する関数
    token_pattern=None,                # デフォルトの英語用正規表現を無効化
    lowercase=False                    # 日本語の小文字化を防ぐ
)
```


## 演習課題

### 演習1_1
以下の文書Aについて、 出現頻度上位3つの用語を示してください。
なお、抽出する用語は、**代名詞・名詞・固有名詞・形容詞・動詞** とする。 

文書A
```text
私は、昨日の午後に大学の図書館に行きました。
図書館には多くの学生が集まり、図書館のカフェでは友人と話す人の姿も見られました。  
私は、まず図書館の新刊コーナーで本を探し、その後で図書館の静かな閲覧室で勉強をしました。  
図書館は勉強や研究にとても役立つ場所だと、私は改めて感じました。  
```

<details>
<summary>解答例</summary>

```python
import spacy
from collections import Counter
import pandas as pd

nlp = spacy.load("ja_ginza")

# 対象文書
texts = ["""
私は、昨日の午後に大学の図書館に行きました。
図書館には多くの学生が集まり、図書館のカフェでは友人と話す人の姿も見られました。  
私は、まず図書館の新刊コーナーで本を探し、その後で図書館の静かな閲覧室で勉強をしました。  
図書館は勉強や研究にとても役立つ場所だと、私は改めて感じました。
"""
]

# 対象の品詞（代名詞、名詞、固有名詞、形容詞、動詞）
target_pos = ("PRON", "NOUN", "PROPN", "ADJ", "VERB")

# ストップワード
stop_words = ["\n", " ", "　"]

# 文書ごとの頻度辞書を保存
doc_counters = []

doc_index = 0
for text in texts:
    doc_index += 1
    doc = nlp(text)

    # 単語リストを作成
    terms = []
    for token in doc:
        if (token.pos_ in target_pos) and (token.text not in stop_words):
            terms.append(token.lemma_)

    # 出現回数をカウント
    tf = Counter(terms)
    doc_counters.append(tf)

# 文書ごとに出現頻度TOP3を表示
for tf in doc_counters:
  print(tf.most_common(3))
```

出力
```text
[('図書館', 6), ('私', 3), ('勉強', 2)]
```
</details>


---

### 演習1_2
以下の3つの文書A~Cについて **TF-IDF 上位の用語1つ** を示してください。  
なお、抽出する品詞は**名詞・固有名詞**とする。

文書A
```text
朝、駅前のカフェで友人と待ち合わせた後、大学の図書館へ向かった。  
図書館の閲覧室は静かで、閲覧室では学生が本を読む姿が多く見られた。  
```

文書B
```text
夕方、駅に近い図書館で本を確認してから、カフェに移動した。  
カフェでは焙煎した豆やバリスタの説明があり、カフェの店内は混雑していた。  
```

文書C
```text
朝の駅は快速の案内で人が流れ、駅のホームでは乗換の表示が目立った。  
改札の前でも駅の混雑が続き、友人と合流するのが大変だった。  
```

<details>
<summary>解答例</summary>

```python
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

nlp = spacy.load("ja_ginza")

# 対象文書
texts = [
"""
朝、駅前のカフェで友人と待ち合わせた後、大学の図書館へ向かった。  
図書館の閲覧室は静かで、閲覧室では学生が本を読む姿が多く見られた。
""",
"""
夕方、駅に近い図書館で本を確認してから、カフェに移動した。  
カフェでは焙煎した豆やバリスタの説明があり、カフェの店内は混雑していた。  
""",
"""
朝の駅は快速の案内で人が流れ、駅のホームでは乗換の表示が目立った。  
改札の前でも駅の混雑が続き、友人と合流するのが大変だった。  
"""
]

# 対象の品詞
target_pos = ("NOUN", "PROPN", "ADJ")
stop_words = ["\n", " ", "　"]

# 形態素解析＋空白区切り化
processed = []
for text in texts:
    doc = nlp(text)
    terms = []
    for token in doc:
        if token.pos_ in target_pos and token.lemma_ not in stop_words:
            terms.append(token.lemma_)
    processed.append(" ".join(terms))

# TF-IDF 計算
def split_by_space(text):
    return text.split()   # 空白で分割してリストにする

vectorizer = TfidfVectorizer(tokenizer=split_by_space, token_pattern=None, lowercase=False)
X = vectorizer.fit_transform(processed)

# DataFrameに変換
df_tfidf = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# 各文書ごとに TF-IDF 上位1語を取得
for i, row in df_tfidf.iterrows():
    top_term = row.idxmax()      # TF-IDFが最も大きい単語
    top_value = row.max()        # その値
    print(f"文書{i+1} の上位1語：{top_term}（TF-IDF={top_value:.3f}）")

```

出力
```text
文書A の上位1：室（TF-IDF=0.458）
文書B の上位1：カフェ（TF-IDF=0.660）
文書C の上位1：駅（TF-IDF=0.557）
```

**補足**
Google Colab では、DataFrame の表示はデフォルトで20列まで。
一時的に、すべての列を表示したい場合は、以下を実行する。

```python
import pandas as pd

# すべての列を表示
pd.set_option("display.max_columns", None)
```

上記をもとに、df_tfidf の中身を確認してみる。

```python
pd.set_option("display.max_columns", None)

df_tfidf
```

出力

|    |   カフェ |   バリスタ |   ホーム |     乗換 |       人 |       前 |     友人 |   図書館 |     多い |     大変 |     大学 |       姿 |     学生 |       室 |    店内 |       後 |     快速 |     改札 |       朝 |       本 |     案内 |     混雑 |     表示 |    説明 |      豆 |    近い |     閲覧 |     静か |       駅 |     駅前 |
|---:|---------:|-----------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|--------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|--------:|--------:|--------:|---------:|---------:|---------:|---------:|
|  0 | 0.174253 |    0       | 0        | 0        | 0        | 0        | 0.174253 | 0.348506 | 0.229122 | 0        | 0.229122 | 0.229122 | 0.229122 | 0.458243 | 0       | 0.229122 | 0        | 0        | 0        | 0.174253 | 0        | 0        | 0        | 0       | 0       | 0       | 0.458243 | 0.229122 | 0        | 0.229122 |
|  1 | 0.660265 |    0.28939 | 0        | 0        | 0        | 0        | 0        | 0.220088 | 0        | 0        | 0        | 0        | 0        | 0        | 0.28939 | 0        | 0        | 0        | 0        | 0.220088 | 0        | 0        | 0        | 0.28939 | 0.28939 | 0.28939 | 0        | 0        | 0.220088 | 0        |
|  2 | 0        |    0       | 0.244091 | 0.244091 | 0.244091 | 0.244091 | 0.185638 | 0        | 0        | 0.244091 | 0        | 0        | 0        | 0        | 0       | 0        | 0.244091 | 0.244091 | 0.244091 | 0        | 0.244091 | 0.244091 | 0.244091 | 0       | 0       | 0       | 0        | 0        | 0.556913 | 0        |


</details>


