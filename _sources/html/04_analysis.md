# 感情分析

```{admonition} 今回のゴール
- 感情分析の目的と応用場面を理解する
- 辞書ベースとBERTベースの感情分析の違いを理解する
- 感情スコアの計算方法を体験する
- 多クラス感情分析を知る
```

---

## 感情分析とは

**感情分析（Sentiment Analysis）** は、テキストデータから「ポジティブ」「ネガティブ」などの感情を自動的に判定する技術。  
SNS・レビュー・ニュース・発言ログなどの大量テキストから、人々の意見・印象・反応を可視化することができる。

---

### 感情分析の応用場面

| 分野 | 応用例 |
|------|--------|
| SNS分析 | Twitterやブログの投稿から人・組織への評価を調べる |
| マーケティング | 新商品に対するポジティブ・ネガティブ比率を分析 |
| 教育 | 学生の授業アンケートから心理傾向を抽出 |
| 政治・社会 | 世論や選挙関連の意見分析 |
| 顧客対応 | 問い合わせメールの感情を自動判別して優先対応 |

---

## 感情スコアの出し方

感情スコア（sentiment score）は、文章の中のポジティブ語・ネガティブ語の出現状況を基に算出する。  
代表的な方法は以下の2つ。

**① 辞書ベース（ルールベース）**
- あらかじめ用意された「感情辞書」を使う
- 各単語にスコア（例：喜 +1.0、悲 -1.0）が登録されている
- 文書内の単語を調べて合計する

**② 機械学習・深層学習ベース（例：BERT）**
- 文の意味全体をモデルが学習してスコア化する
- 文脈に依存する表現も理解できる

---

### ① 辞書ベース感情分析の具体例

**例文**
> この映画はとても面白かったが、少し退屈な場面もあった。

---

**ステップ1：文を単語に分割する**  
形態素解析を行い、感情に関連する単語を抽出する。

| 抽出単語 | 備考 |
|-----------|------|
| 面白い | ポジティブ表現 |
| 退屈 | ネガティブ表現 |

---

**ステップ2：辞書から感情スコアを参照**

| 単語 | 感情極性 | スコア |
|------|-----------|--------|
| 面白い | ポジティブ | +2 |
| 退屈 | ネガティブ | -1 |

---

**ステップ3：スコアの集計**

文章全体の感情スコアは次のように計算する。

> この映画はとても<u>面白</u>かったが、少し<u>退屈</u>な場面もあった。

総合スコア：
\[
(+2) + (-1) = +1
\]

**→ 結果：ややポジティブ**

---

#### 複数文での平均スコア計算例

**例：あるユーザのレビュー投稿**
> この映画は最高だった。  
> ストーリーは退屈だった。  
> 映像は素晴らしいが、内容は少し残念だ。  
> 音楽は普通だった。  

**ポジネガ辞書**

| 感情の種類 | 単語例 | スコア |
|-------------|--------|--------|
| 強いポジティブ | 最高、感動、素晴らしい | +2 |
| ややポジティブ | 良い、楽しい、嬉しい | +1 |
| ニュートラル（中立） | 普通、まあまあ | 0 |
| ややネガティブ | 退屈、残念、不安 | −1 |
| 強いネガティブ | 最悪、嫌い、ひどい | −2 |


**計算例**
| 文 | 総スコア |
|----|-----------|
| この映画は<u>最高</u>だった。 | +2 |
| ストーリーは<u>退屈</u>だった。 | -1 |
| 映像は<u>素晴らしい</u>が、内容は少し<u>残念</u>だ。 | +1 |
| 音楽は<u>普通</u>だった。 | 0 |

平均スコア：
\[
((+2) + (-1) + (+1) + (0)) / 4 = 0.5
\]

**→ 全体としてはややポジティブな傾向であると判断できる。**

---

##### プログラム例（ポジネガ辞書）

**ライブラリインストール（必要に応じて）**  
```python
!pip -q install -U spacy ginza ja-ginza
```

**プログラム**
```python
import spacy

nlp = spacy.load("ja_ginza")

# 感情スコア付き辞書（見出し語）
word_scores = {
    "最高": 2,
    "感動": 2,
    "素晴らしい": 2,
    "良い": 1,
    "楽しい": 1,
    "嬉しい": 1,
    "普通": 0,
    "まあまあ": 0,
    "退屈": -1,
    "残念": -1,
    "不安": -1,
    "最悪": -2,
    "嫌い": -2,
    "ひどい": -2,
}

texts = [
    "この映画は最高だった。",
    "ストーリーは退屈だった。",
    "映像は素晴らしいが、内容は少し残念だ。",
    "音楽は普通だった。",
]

scores = []

for text in texts:
    doc = nlp(text)
    score = 0
    for token in doc:
        lemma = token.lemma_
        if lemma in word_scores:
            score = score + word_scores[lemma]
    scores.append(score)
    print(text, "→ スコア:", score)

total = 0
for s in scores:
    total = total + s
average = total / len(scores)
print("平均スコア:", average)
```

---

**出力**

```
この映画は最高だった。 → スコア: 2
ストーリーは退屈だった。 → スコア: -1
映像は素晴らしいが、内容は少し残念だ。 → スコア: 1
音楽は普通だった。 → スコア: 0
平均スコア: 0.5
```

---

### ② 機械学習・深層学習ベース（例：BERT）

BERTは、Googleが開発した大規模言語モデルであり、単語の文脈を深く理解することができる。  
BERTベースの感情分析では、テキストを文脈ごとにベクトル化し、ポジティブ・ネガティブなどのクラスに分類する。

#### プログラム例（RoBERTaモデルによるポジネガスコア）

**RoBERTaモデルのインストール**  
今回は、Hugging Face の公開モデルを使って文章の感情（ポジティブ／ネガティブ／ニュートラル）を判定する  
[cardiffnlp/twitter-xlm-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment)

```python
!pip install -q transformers sentencepiece -U torch torchvision torchaudio
```

**プログラム**
```python
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from tqdm.auto import tqdm

# モデルの読み込み
model_id = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tok = AutoTokenizer.from_pretrained(model_id)
mdl = AutoModelForSequenceClassification.from_pretrained(model_id)

# パイプライン作成
pipe = TextClassificationPipeline(
    task="sentiment-analysis",
    tokenizer=tok,
    model=mdl,
    top_k=None,
    truncation=True,
    max_length=512
)

texts = [
    "この映画はとても面白かった。映像も綺麗だった。",
    "期待していたけど、ちょっと退屈だった。",
    "最悪の出来でした。二度と見たくない。",
    "普通かな。特に印象に残らなかった。",
    "主演俳優の演技が素晴らしかった！"
]

# 推論とDataFrame化
rows = []
for t in tqdm(texts, total=len(texts), unit="件"):
    results = pipe(t)[0]  # 各ラベルごとのスコア
    row = {"文章": t}
    for r in results:
        label = r["label"]
        score = round(float(r["score"]), 3)
        row[label] = score
    rows.append(row)

df_out = pd.DataFrame(rows)
df_out
```

**出力**
| 文章                      | positive | neutral | negative |
| :---------------------- | -------: | ------: | -------: |
| この映画はとても面白かった。映像も綺麗だった。 |    0.930 |   0.052 |    0.018 |
| 期待していたけど、ちょっと退屈だった。     |    0.053 |   0.141 |    0.806 |
| 最悪の出来でした。二度と見たくない。      |    0.026 |   0.032 |    0.941 |
| 普通かな。特に印象に残らなかった。       |    0.065 |   0.244 |    0.691 |
| 主演俳優の演技が素晴らしかった！        |    0.919 |   0.051 |    0.030 |
---

## 発展：多クラス感情分析

これまでの講義では、「ポジティブ」「ネガティブ」といった**2値分類の感情分析**を扱った。  
しかし、人間の感情には「喜び」「怒り」「悲しみ」「恐れ」「驚き」など、より多様な種類がある。  
このような複数の感情カテゴリに分類する分析を **「多クラス感情分析（multi-class sentiment analysis）」** という。

---

### 多クラス感情分析とは
多クラス感情分析では、文章をあらかじめ定義された**複数の感情カテゴリのいずれか（またはいくつか）**に分類する。  
分類対象となる感情は研究や応用分野によって異なるが、代表的な例としては以下がある。

| 感情カテゴリ | 英語表現 | 説明 |
|---------------|-----------|------|
| 喜び | Joy | ポジティブで幸福感を含む感情 |
| 悲しみ | Sadness | 喪失感や落胆 |
| 期待 | Anticipation | 将来への期待や希望 |
| 驚き | Surprise | 予想外の出来事への反応 |
| 怒り | Anger | 不満や攻撃性を含む感情 |
| 恐れ | Fear | 不安や危険への反応 |
| 嫌悪 | Disgust | 拒絶や嫌悪の感情 |
| 信頼 | Trust | 他者への安心・信頼感 |

これらの8種類は、心理学者ロバート・プルチックの「感情の輪（Wheel of Emotions）」に基づく分類であり、自然言語処理でもよく利用される。

---

### BERTによる多クラス感情分析
多クラス感情分析には、近年では **BERT** や **RoBERTa** などの事前学習済み言語モデルがよく使われる。  
BERTは大量のテキストから文の意味を深く学習しており、文の特徴を自動的に抽出して分類タスクに応用できる。

このモデルを使うと、単語ベースの特徴抽出や辞書ベースの手法に比べて、文脈に基づいた精度の高い感情分類が可能になる。  
今回は、日本語向けに学習された多クラス感情モデルを用いる。

---

### プログラム例（BERTによる多クラス感情分析）

以下は、**8クラス感情分析**を行うシンプルなPythonサンプル。  
事前学習済みモデルとして `koshin2001/Japanese-to-emotions` を使用する。

**ライブラリのインストール**  
Google Colab 環境では、以下２つを順に実行する（クラッシュ回避のため）
```python
!pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --quiet
```
```python
!pip install transformers sentencepiece fugashi unidic-lite pandas==2.2.2 --quiet
```

**プログラム**
```python
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from tqdm.auto import tqdm

# モデル読み込み
model_id = "koshin2001/Japanese-to-emotions"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

# BERT処理
if "token_type_ids" in getattr(tokenizer, "model_input_names", []):
    names = []
    i = 0
    while i < len(tokenizer.model_input_names):
        n = tokenizer.model_input_names[i]
        if n != "token_type_ids":
            names.append(n)
        i = i + 1
    tokenizer.model_input_names = names

# 8クラスの日本語ラベル
custom_labels = ["喜び", "悲しみ", "期待", "驚き", "怒り", "恐れ", "嫌悪", "信頼"]

# 感情分析パイプライン
pipe = TextClassificationPipeline(
    task="text-classification",
    tokenizer=tokenizer,
    model=model,
    truncation=True,
    max_length=512
)

texts = [
    "明日の発表が楽しみです！",
    "最悪の結果になってしまった…",
    "信じられない、どうしてこんなことに。",
    "先生に怒られて腹が立った。",
    "正直ちょっと不安です。",
    "あなたのことを信頼しています。"
]

# 推論とDataFrame化
rows = []
for t in tqdm(texts, desc="感情分析中", unit="件"):
    results = pipe(t, top_k=None)
    row = {"文章": t}

    for r in results:
        label = r["label"]
        if label.startswith("LABEL_"):
            try:
                idx = int(label.split("_")[1])
                if 0 <= idx < len(custom_labels):
                    label = custom_labels[idx]
            except:
                pass
        row[label] = round(float(r["score"]), 3)

    rows.append(row)

# ===== DataFrame化 =====
df_out = pd.DataFrame(rows)
cols = ["文章"] + custom_labels
df_out = df_out[cols]
df_out

```

---

**出力**

| 文章                 | 喜び    | 悲しみ   | 期待    | 驚き    | 怒り    | 恐れ    | 嫌悪    | 信頼    |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 明日の発表が楽しみです！       | 0.321 | 0.003 | 0.587 | 0.014 | 0.001 | 0.004 | 0.002 | 0.061 |
| 最悪の結果になってしまった…     | 0.003 | 0.606 | 0.004 | 0.023 | 0.033 | 0.196 | 0.132 | 0.003 |
| 信じられない、どうしてこんなことに。 | 0.012 | 0.241 | 0.016 | 0.319 | 0.030 | 0.247 | 0.117 | 0.011 |
| 先生に怒られて腹が立った。      | 0.006 | 0.413 | 0.006 | 0.028 | 0.212 | 0.057 | 0.249 | 0.005 |
| 正直ちょっと不安です。        | 0.006 | 0.104 | 0.010 | 0.052 | 0.008 | 0.767 | 0.083 | 0.012 |
| あなたのことを信頼しています。    | 0.218 | 0.081 | 0.215 | 0.025 | 0.024 | 0.023 | 0.035 | 0.409 |

---

## 補足：複数文を含んだ文書の感情スコア算出について

複数文を含んだ文書において、感情スコアをどのように算出するかは、  
以下の2つの考え方がある。  

**① 一文単位でスコアを算出して平均する方法**  
**② 複数文をまとめて1つのテキストとして扱う方法**  

---

### ① 一文単位でスコアを算出して平均する方法

文書を文ごとに分割し（句点「。」などで区切る）、  
各文ごとにポジティブ・ネガティブスコアを算出して平均する。

**例**  
> 「映画の前半は退屈だった。後半は最高だった！」  

→ 「映画の前半は退屈だった」：ネガティブ（-1） + 「後半は最高だった！」：ポジティブ（+1） / 2  
**→ 平均スコア： 0（中立的な印象）**

**メリット**
- 文ごとの感情のばらつきを反映できる  
- 複数の意見が混在する投稿に対応できる  
- SNSや口コミ分析では一般的な方法

**デメリット**
- 文分割の精度に影響を受ける  
- 感情の強さ（重み）を考慮しにくい

**適している場面**
- SNS投稿・レビューなど複数意見が混じるケース  
- 辞書ベース手法やシンプルな機械学習モデルを使う場合

---

### ② 複数文をまとめて1つのテキストとして扱う方法

文書全体を1つの文章として入力し、まとめて感情スコアを算出する。  
文単位ではなく、全体のトーンや文脈を重視する。

**例**  
> 「最初はどうなるか不安だった。だけど結果的にはすごく満足している。」  

→ 不安（ネガティブ）＋ 満足（ポジティブ）だが、  
**→ 全体では「ポジティブ寄りの感情」と判断される。**  
  
モデルが前後の文脈を理解して、最終的な印象を出せる。

**メリット**
- BERTなど文脈を理解するモデルではこちらが有効  
- 文脈全体のトーンや感情の流れを把握できる  
- 文分割不要でシンプルに実装できる

**デメリット**
- 部分的な強い感情が平均化されやすい  
- 投稿が長すぎるとモデルのトークン長を超える場合がある（例：512字）

**適している場面**
- レビューやニュース記事など全体印象を評価したい場合  
- BERT / RoBERTa などの文脈理解モデルを利用する場合

---

<!-- **ポイント** 
> - 「文単位平均」は構成のばらつきを反映しやすい  
> - 「全体一括処理」は文脈理解モデルで効果的  
> - 分析目的とモデル特性に応じて選択することが重要
 -->
