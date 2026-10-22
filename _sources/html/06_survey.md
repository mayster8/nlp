# アンケート分析（演習6）

```{admonition} 今回のゴール
- 自由記述アンケートの回答に**ポジネガ値**を付与し、全体の傾向をつかむ
- **ネガティブな回答だけ**をLDAで分析し、不満の「テーマ」を見つける
- 分析結果から**改善提案**を1〜2点まとめる
```

## 演習の流れ

大学の施設に関する学生アンケート（自由記述）が2種類ある。**どちらか1つ**を選んで分析する。

| データ | URL |
| :-- | :-- |
| アンケートA | http://ogawa.isslab.org/nlp/data/data-surveyA.csv |
| アンケートB | http://ogawa.isslab.org/nlp/data/data-surveyB.csv |

- 列は `回答` の1列のみ（1行が1人分の回答）。文字コードは `utf-8-sig`
- 回答は平均13〜15文字ほどの**短文**。繰り返しなどの明らかなノイズは取り除いてある

```{note}
K-meansは今回のデータには向かない。回答が短く、1回答あたりの名詞が2つ程度しかないため、文書同士で共通する語が少なく、**1つの巨大なクラスタ＋小さなクラスタ**に偏りやすい。LDAは「トピックごとの語の出やすさ」を推定するので、短文でもおおまかなテーマを取り出せる。
```

1. **ポジネガ分析**：各回答に「ポジネガ値」を付与する
2. **LDA**：ネガティブな回答だけを取り出して、トピックを分析する
3. **まとめ**：トピックに名前を付け、改善提案を書く

## ステップ1：ポジネガ値の付与

「ポジネガ値 ＝ positiveの確率 − negativeの確率」とする（−1〜+1。+1に近いほどポジティブ、−1に近いほどネガティブ）。

**プログラム**
```python
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from tqdm import tqdm

# アンケートの選択（A または B）
name = "A"
url = f"http://ogawa.isslab.org/nlp/data/data-survey{name}.csv"
df = pd.read_csv(url, encoding="utf-8-sig")
texts = df["回答"].to_list()
print(len(texts), "件")

# ポジネガ分析モデル（授業で使ったもの）
model_id = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tok = AutoTokenizer.from_pretrained(model_id)
mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
pipe = TextClassificationPipeline(task="sentiment-analysis", tokenizer=tok, model=mdl,
                                  top_k=None, truncation=True, max_length=512)

# 各回答に「ポジネガ値」を付与：スコア = positive確率 − negative確率（-1〜+1）
scores = []
for t in tqdm(texts, unit="件"):
    r = {x["label"].lower(): x["score"] for x in pipe(t)[0]}
    scores.append(r["positive"] - r["negative"])
df["ポジネガ値"] = scores

print(df["ポジネガ値"].describe().round(2))
print(df.sort_values("ポジネガ値").head(5).to_string())
print(df.sort_values("ポジネガ値").tail(5).to_string())
```

**確認すること**
- 平均値はプラスか、マイナスか
- 最もネガティブな回答／最もポジティブな回答を読んで、判定は妥当か
- ヒストグラム（`df["ポジネガ値"].hist(bins=20)`）で分布を確認する

```{tip}
短文では「特にない」「分からない」のような回答もポジティブ／ネガティブに振られることがある。判定が怪しい回答を見つけたら、レポートに書いておくとよい。
```

## ステップ2：ネガティブ回答のLDA

ステップ1の続きとして実行する（`df` を使う）。

**プログラム**
```python
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ネガティブな回答だけを取り出す（しきい値は自分で決める）
th = -0.3
neg = df[df["ポジネガ値"] <= th].reset_index(drop=True)
print(f"ネガティブ回答: {len(neg)} 件 / {len(df)} 件")

# 形態素解析：名詞・固有名詞（原形）。「お」「こと」など意味の薄い語は除く
nlp = spacy.load("ja_ginza")
target_pos = ("NOUN", "PROPN")
stop_words = ["\n", " ", "　", "お", "こと", "ところ", "感", "時", "前", "数", "的", "人", "もの", "方", "気", "ため", "よう", "点"]

def tokenize(text):
    doc = nlp(text)
    return " ".join(t.lemma_ for t in doc if t.pos_ in target_pos and t.lemma_ not in stop_words)

processed = [tokenize(t) for t in neg["回答"]]
neg["tokens"] = processed
neg = neg[neg["tokens"] != ""].reset_index(drop=True)   # 名詞が無い回答は除く
print("LDAに使う回答:", len(neg), "件")

# Bag of Words（短文なので min_df=3：3回答以上に出る語だけ使う）
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), token_pattern=None,
                             lowercase=False, min_df=3)
X = vectorizer.fit_transform(neg["tokens"])
print("語彙数:", X.shape[1])

# LDA（Kは自分で変えて比べる）
K = 3
lda = LatentDirichletAllocation(n_components=K, max_iter=200, learning_method="batch",
                                doc_topic_prior=0.1, topic_word_prior=0.1, random_state=0)
lda.fit(X)

# 各トピックの上位語
words = vectorizer.get_feature_names_out()
for i, topic in enumerate(lda.components_):
    print(f"トピック{i}:", ", ".join(words[j] for j in topic.argsort()[::-1][:8]))

# 各トピックの代表的な回答（そのトピックの割合が高い回答）
theta = lda.transform(X)
neg["トピック"] = theta.argmax(axis=1)
for i in range(K):
    print(f"\n【トピック{i}】{(neg['トピック']==i).sum()}件")
    sub = neg[neg["トピック"] == i].copy()
    sub["割合"] = theta[sub.index, i]
    print(sub.sort_values("割合", ascending=False)["回答"].head(5).to_string(index=False))
```

**自分で試すこと**
- しきい値 `th` を変える（−0.1、−0.5 など）と、ネガティブ回答の件数とトピックはどう変わるか
- トピック数 `K` を 3、4、5 と変えて比べる。「トピックが細かすぎる」「複数のテーマが混ざる」ときはKを調整する
- 各トピックの上位語と代表的な回答を読み、**トピックに名前を付ける**（例：「席・コンセントの不足」）

```{note}
LDAは実行のたびにトピックの番号が入れ替わることがある（`random_state` を固定すれば同じ結果になる）。**番号ではなく上位語から意味を解釈する**のは人間の役割。
```

## ステップ3：結果のまとめ

以下を1枚の画像（表またはグラフ）にして提出する。

| 項目 | 内容 |
| :-- | :-- |
| 使ったデータ | アンケートA／B、回答数 |
| ポジネガ | 平均値、ネガティブ回答の件数と割合（しきい値も明記） |
| トピック | K、各トピックの名前・上位語・件数 |
| 代表的な回答 | 各トピックから1〜2件 |
| 改善提案 | 1〜2点。根拠となるトピック・回答を添える |

```{admonition} 考察のヒント
- ポジティブな回答が多いのに、ネガティブな回答に多くの不満が集まるテーマはあるか
- 件数が多いトピックと、少ないが深刻なトピックのどちらを優先して改善するか
- 短文・少数のデータで、トピックの解釈にどんな限界があるか
```
