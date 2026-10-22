# 環境構築

```{admonition} 今回のゴール
- Gooble Colab ノートブックの基本操作を理解する  
- Python 基礎を復習する（データ型、リスト・辞書、for/関数）  
- spaCy / GiNZA を導入し、日本語文を解析できるようにする  
```

---

## Google Colab の使い方

### ノートブックを作成
1. **Google Drive** を開く  
2. 「新規」→「その他」→「Google Colaboratory」 を選択  
   - 無い場合は [Colab](https://colab.research.google.com/) から「新しいノートブック」を作成  
3. タイトルを変更（例：`自然言語処理_環境構築`）  

### 基本操作
- セル実行：`Shift + Enter`  
- セルの種類：コードセル / テキストセル  
- 保存：Driveに自動保存  

### 便利な機能
- その場でライブラリをインストール可能  
```bash
  !pip install パッケージ名
```

## Python 基礎の復習

> 下のコードを Google Colab に貼って実行してみましょう。

### データ型と演算
```python
x = 10
y = 3.14
name = "text"
flag = True

print(x, y, name, flag)
print(x + 5, y * 2, name+"mining", flag, not(flag))
```

<details>
<summary>出力例を表示する</summary>

```text
10 3.14 text True
15 6.28 textmining True False
```
</details>

### リストと辞書
```python
fruits = ["apple", "banana", "orange"]   # list
scores = {"math": 80, "english": 90}     # dict

fruits.append("grape")
scores["science"] = 85

print(fruits)
print(scores)
print(f"英語: {scores['english']}")
```

<details>
<summary>出力例を表示する</summary>

```text
['apple', 'banana', 'orange', 'grape']
{'math': 80, 'english': 90, 'science': 85}
英語: 90
```
</details>

### 条件分岐と for ループ
```python
nums = [1, 2, 3, 4, 5]

for n in nums:
    if n % 2 == 0:
        print(n, "は偶数")
    else:
        print(n, "は奇数")
```

<details>
<summary>出力例を表示する</summary>

```text
1 は奇数
2 は偶数
3 は奇数
4 は偶数
5 は奇数
```
</details>

### 関数
```python
def add(a, b):
    return a + b

print(add(5, 7))
```
<details>
<summary>出力例を表示する</summary>

```text
12
```
</details>

### リスト内包表記（発展）

リストを作るとき、for文を1行で書ける便利な方法です。

```python
nums = [1, 2, 3]

# 普通のfor文で2倍のリストを作る
doubled = []
for n in nums:
    doubled.append(n * 2)

# 内包表記を使うと1行で書ける
doubled_comp = [n * 2 for n in nums]

print("2倍:", doubled)
print("2倍（内包表記）:", doubled_comp)
```

<details>
<summary>出力例を表示する</summary>

```text
2倍: [2, 4, 6]
2倍（内包表記）: [2, 4, 6]
```
</details>

---

## spaCy / GiNZA の導入

### spaCy / GiNZA とは？

#### spaCy（スペイシー） とは
- **自然言語処理（NLP）のための Python ライブラリ**
- 高速かつ実用的な処理が可能で、研究よりも「実務やアプリ開発」で広く使われている  
- 主な機能：
  - トークン化（文を単語に分割）
  - 品詞タグ付け（名詞、動詞、形容詞など）
  - 依存構造解析（主語・述語・目的語の関係を解析）
  - 固有表現抽出（人名、地名、日付など）

**英語を中心に多数の言語モデルが提供されており、簡単に導入できるのが強み**

---

#### GiNZA（ギンザ） とは
- **spaCy の日本語用ライブラリ／言語モデル**
- 日本語は英語と異なり「単語の区切りが曖昧」なので、形態素解析が必要  
- GiNZA は日本語文章を対象にした **高精度の形態素解析・依存構造解析** を実現  
- 主な特徴：
  - 名詞・動詞などの品詞情報を取得可能
  - 活用形を原形（lemma）に変換
  - 文の依存関係を解析して構造を明らかにする

---

#### spaCy と GiNZA の関係
- **spaCy**：自然言語処理の「本体」（フレームワーク）  
- **GiNZA**：その中で使う「日本語用の言語モデル」  
- イメージ：
  - spaCy = 車の本体
  - GiNZA = 日本語を走らせるための専用エンジン

### インストール
```python
!pip -q install spacy ginza ja-ginza
```

### 日本語モデルをロード
```python
import spacy
nlp = spacy.load("ja_ginza")

doc = nlp("私は昨日、東京で美味しいラーメンを食べた。")

# 形態素を抽出してリストに追加
tokens = []
for t in doc:
    tokens.append(t.text)

print(tokens)  # トークン化の確認
```

出力例：
```
['私', 'は', '昨日', '、', '東京', 'で', '美味しい', 'ラーメン', 'を', '食べ', 'た', '。']
```

### 形態素解析の詳細表示
```python
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.dep_)
```

出力例：
```
私       私       PRON   nsubj
昨日     昨日     NOUN   obl
東京     東京     PROPN  obl
美味しい 美味しい ADJ    amod
ラーメン ラーメン NOUN   obj
食べ     食べ     VERB   ROOT
た       た       AUX    aux
。       。       PUNCT  punct
```


---

#### 各列の意味

1. **token.text**  
   → 文に現れた形（表層形）  
   例：「私」「昨日」「東京」

2. **token.lemma_**  
   → 原形（辞書形）  
   例：「食べた」→「食べ」、「美味しかった」→「美味しい」

3. **token.pos_**  
   → 品詞（大分類）  
   - `PRON` = 代名詞  
   - `NOUN` = 名詞  
   - `PROPN` = 固有名詞  
   - `ADJ` = 形容詞  
   - `VERB` = 動詞  
   - `AUX` = 助動詞  
   - `PUNCT` = 記号

4. **token.dep_**  
   → 文中での役割（係り受け関係）  
   - `nsubj` = 主語  
   - `obl` = 修飾語（副詞的）  
   - `amod` = 形容詞による修飾  
   - `obj` = 目的語  
   - `ROOT` = 文の中心（述語）  
   - `aux` = 助動詞  
   - `punct` = 句読点

---

#### 出力の解釈例

- **私 (PRON, nsubj)** → 主語（誰が？）  
- **昨日 (NOUN, obl)** → 修飾語（いつ？）  
- **東京 (PROPN, obl)** → 修飾語（どこで？）  
- **美味しい (ADJ, amod)** → 「ラーメン」を修飾する形容詞  
- **ラーメン (NOUN, obj)** → 目的語（何を？）  
- **食べ (VERB, ROOT)** → 文の中心（述語）  
- **た (AUX, aux)** → 助動詞（過去形を表す）  
- **。 (PUNCT, punct)** → 句点  

---

#### 名詞だけを抽出する
```python
text = "自然言語処理の授業で、PythonとGiNZAを学びます。"
doc = nlp(text)

nouns = []  # 名詞を入れるリスト
for t in doc:
    if t.pos_ == "NOUN" or t.pos_ == "PROPN":
        nouns.append(t.lemma_)

print(nouns)
```


---

## トラブルシューティング

- `OSError: [E050] Can't find model 'ja_ginza'`  
  → インストールセルを実行後、**ランタイムを再起動**（ランタイム → 再起動）  

- `ModuleNotFoundError: No module named 'spacy'`  
  → `!pip install -U spacy` を実行してから再度 `import spacy`  

- 日本語が文字化け（ローカル環境で図を描く場合）  
  → 日本語フォントを設定する（Colab では通常不要）

---



## まとめ
- Gooble Colab で簡単に Python を実行できる  
- Python 基礎（変数・リスト・辞書・for・関数）を復習  
- GiNZA を導入して日本語形態素解析が可能になった  

---

## 演習課題
 
### 課題1
- 次の文を GiNZA で解析し、単語ごとに分割して出力してみましょう

```python
import spacy
nlp = spacy.load("ja_ginza")

doc = nlp("私は自然言語処理を学んでいます。")
words = []
for t in doc:
    words.append(t.text)
print(words)
```

### 課題2
- 品詞を表示してみましょう

```python
doc = nlp("私は自然言語処理を学んでいます。")
for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

### 課題3
- 以下の関数は「名詞」「固有名詞」だけをリストで取り出すプログラムです。

```python
def extract_nouns(text):
    doc = nlp(text)
    words = []
    for t in doc:
        if t.pos_ in ("NOUN", "PROPN"):
            words.append(t.text)
    return words

sentence = "自然言語処理の授業で、PythonとGiNZAを学びます。"
print(extract_nouns(sentence))
```

- 上記を修正して、 **「動詞」** のみをリストで取り出してください


### 課題4

- 以下の文章を形態素解析すると、どのような単語と品詞に分割されるでしょうか。

```text
すもももももももものうち
うらにわにはにわにわとりがいる
外国人参政権

```

- ２つの文章について、分割された単語と品詞を示してください。
 

---
