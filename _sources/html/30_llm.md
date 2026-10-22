# 大規模言語モデル（LLM）

```{admonition} 今回のゴール
- 自然言語処理（NLP）の進化の流れを理解する
- 単語の意味表現（Word2Vec）、文脈理解（BERT）、文章生成（GPT）の技術について理解する
- Word2Vec, BERT, GPT のプログラムを実装し、その動作と応用範囲について学ぶ

```

---

## Word2Vec

### Word2Vec演習：類義語の検索

```python
!pip install -q gensim
```

**プログラム**  

```python
from gensim.models import KeyedVectors

url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ja.300.vec.gz"
model = KeyedVectors.load_word2vec_format(url, binary=False, limit=100000)

print("モデル読み込み完了。単語数：", len(model))
```

```python
# 類義語の検索
word = "猫"

print(f"「{word}」に近い単語：")
for word, sim in model.most_similar(word, topn=20):
    print(f"  {word:10s} 類似度: {sim:.3f}")
```

**出力**  

```text
「猫」に近い単語：
  ネコ         類似度: 0.806
  ねこ         類似度: 0.727
  子猫         類似度: 0.720
  仔猫         類似度: 0.706
  ニャンコ       類似度: 0.706
  野良猫        類似度: 0.703
  犬          類似度: 0.651
  ミケ         類似度: 0.636
  ゃんこ        類似度: 0.616
  ゃんこの       類似度: 0.604
  猫又         類似度: 0.562
  野良         類似度: 0.561
  ニャンズ       類似度: 0.560
  ウサギ        類似度: 0.556
  ノラ         類似度: 0.555
  シュレーディンガー  類似度: 0.553
  マタタビ       類似度: 0.549
  チンチラ       類似度: 0.547
  ロシアンブルー    類似度: 0.547
  ニャー        類似度: 0.546
```

---

### Word2Vec演習：単語の足し算・引き算

**プログラム**

```python
# 単語の足し算・引き算
# A - B + C
A = "王"
B = "男"
C = "女"

print(f"「{A}」-「{B}」+「{C}」に近い単語：")
for w, s in model.most_similar(positive=[A, C], negative=[B], topn=10):
    print(f"  {w:10s} 類似度: {s:.3f}")
```

**出力**

```text
「王」-「男」+「女」に近い単語：
  王妃         類似度: 0.484
  女王         類似度: 0.466
  遊戯         類似度: 0.458
  陵          類似度: 0.446
  冥          類似度: 0.438
  聖王         類似度: 0.436
  妃          類似度: 0.436
  王様         類似度: 0.432
  太子         類似度: 0.431
  王国         類似度: 0.429
```

---

### Word2Vec演習：意味ベクトルの可視化

```python
!pip -q install japanize_matplotlib
```

**プログラム**

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import japanize_matplotlib

words = ["東京", "大阪", "京都", "ラーメン", "うどん", "そば", "パスタ"]

# url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ja.300.vec.gz"
# model = KeyedVectors.load_word2vec_format(url, binary=False, limit=100000)

vectors = []
for w in words:
  vectors.append(model[w])

pca = PCA(n_components=2)
reduced = pca.fit_transform(vectors)

plt.figure(figsize=(6, 6))
for i, w in enumerate(words):
    plt.scatter(reduced[i, 0], reduced[i, 1])
    plt.text(reduced[i, 0] + 0.01, reduced[i, 1] + 0.01, w)
plt.title("Word2Vecによる意味空間の可視化")
plt.show()
```

**出力**  

<img src=../static/out-llm-word2vec.png>

---

## BERT

### BERT演習：穴埋め予測タスク

```python
!pip install -q fugashi ipadic unidic-lite transformers
```

**プログラム**

```python
from transformers import pipeline, logging
logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore")

unmasker = pipeline("fill-mask", model="izumi-lab/bert-small-japanese")
```

```python
text = "私はカフェで[MASK]を注文しました"
# text = "朝起きたら[MASK]が降っていました。"
# text = "昨日は[MASK]で授業を受けました。"
# text = "昨日は[MASK]で講義を受けました。"
# text = "この映画は本当に[MASK]と思った。"
# text = "学校に行く途中で[MASK]を落としました。"

results = unmasker(text)
print("原文:", text)
print("BERTの予測:")
for r in results:
    print(f"  {r['token_str']}（スコア: {r['score']:.3f}）")
```

**出力**

```text
原文: 私はカフェで[MASK]を注文しました
BERTの予測:
  コーヒー（スコア: 0.075）
  食事（スコア: 0.062）
  料理（スコア: 0.060）
  絵（スコア: 0.026）
  ワイン（スコア: 0.022）
```

---

## Sentence-BERT演習：文章の意味ベクトル

```python
!pip install -q sentence-transformers pyvis
```

**プログラム**

```python
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from pyvis.network import Network
from IPython.display import HTML

# モデルのロード
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
```

```python

sentences = [
    "東京タワーからの夜景は本当にきれいだった。",
    "浅草の仲見世通りは人でいっぱいでした。",
    "京都の清水寺では紅葉が見頃でした。",
    "奈良では鹿と触れ合うことができました。",
    "大阪の道頓堀でたこ焼きを食べました。",
    "札幌の雪まつりは迫力がありました。",
    "博多のラーメンはスープが濃厚で美味しかった。",
    "名古屋の味噌カツは初めて食べたけど最高でした。",
    "沖縄の海は信じられないくらい青かった。",
    "仙台の牛タンは肉厚で柔らかかった。"
]

# sentences = [
# "今日は本当に嬉しい一日だった。",
# "悲しいニュースを見て心が痛んだ。",
# "友達と遊んでとても楽しかった。",
# "仕事で失敗して落ち込んだ。",
# "美しい景色を見て感動した。",
# "試験の結果にがっかりした。",
# "プレゼントをもらって幸せな気分になった。",
# "大切なものをなくしてショックだった。",
# "久しぶりに家族に会えて安心した。",
# "上司に怒られてつらかった。",
# ]


# 文章に連番 id を付ける
ids = list(range(1, len(sentences) + 1))

# sentence-BERT
embeddings = model.encode(sentences)

# 類似度行列を計算
sim = util.cos_sim(embeddings, embeddings)
df_sim = pd.DataFrame(sim, index=ids, columns=ids)

# ネットワーク作成
net = Network(
    height="600px",
    width="100%",
    bgcolor="white",
    notebook=True,
    cdn_resources="in_line"  # Colab用
)

# ノード追加
for i, text in zip(ids, sentences):
    label = text[:20] + "..." if len(text) > 20 else text
    net.add_node(i, label=label, title=text)

# エッジ追加（類似度が閾値以上の場合）
threshold = 0.35
for i in ids:
    for j in ids:
        if i < j:
            score = float(df_sim.loc[i, j])
            if score >= threshold:
                net.add_edge(i, j, value=score, title=f"類似度: {score:.2f}")

# ===== 8. ネットワーク表示（Colab） =====
net.prep_notebook()
html = net.generate_html()
display(HTML(html))
```

**出力**

<img src=../static/out-llm-network.png>

---

### Sentence-BERT演習：類似文の検索

```python
!wget -q https://raw.githubusercontent.com/ids-cv/wrime/refs/heads/master/wrime-ver1.tsv -O wrime-ver1.tsv
```

**プログラム**

```python
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# モデルのロード（日本語対応）
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Twitterデータ読み込み
df = pd.read_csv("wrime-ver1.tsv", sep="\t")

# 文章列を抽出（ランダムサンプリング1000件）
texts = df["Sentence"].dropna().sample(1000, random_state=0).tolist()
# texts = df["Sentence"].dropna().tolist()

# 意味ベクトル化
embeddings = model.encode(texts, convert_to_tensor=True)

# 類似検索関数
def search_similar_sentences(query, top_k=5):
    # 入力文をベクトル化
    query_vec = model.encode(query, convert_to_tensor=True)

    # 類似度を計算
    cos_scores = util.cos_sim(query_vec, embeddings)[0]

    # 上位のスコアを抽出
    top_results = cos_scores.topk(top_k)

    print(f"\n検索クエリ: {query}\n")
    print("===== 類似上位文 =====")
    for score, idx in zip(top_results.values, top_results.indices):
        print(f"スコア: {score:.3f} | {texts[idx]}")
    print("=========================")
```

```python
text = "最近寒くなってきたのでコートを着ます"

search_similar_sentences(text, top_k=10)
```

**出力**

```text
検索クエリ: 最近寒くなってきたのでコートを着ます

===== 類似上位文 =====
スコア: 0.830 | 今日も寒いからね。カイロもってるのかなー。
スコア: 0.737 | てか寒くない？
スコア: 0.703 | おはよーです。風邪よくなったけど乾燥してると咳き込んじゃいます。
スコア: 0.672 | おしゃれ中止！！！！極寒らしいな
スコア: 0.663 | 全てを雪のせいにしていた頃が懐かしい…
スコア: 0.645 | 晴れだから雪溶けると言ったけど、大して晴れてないし、雪もちょっと降ってるから溶けなそう。。。
スコア: 0.640 | 疲れすぎて、気づいたら風邪っぽさふっとんでた＼(^^)／
スコア: 0.625 | 本格的に風邪引くのかなあ。。 やだなあ。。 明日の予定が 死なないといいんだが
スコア: 0.607 | ダウンジャケットがもう暑い、10月の8時。
スコア: 0.601 | マスク冬用の買わないとなー
=========================
```

---

## GPT

### GPT演習：文章生成

```python
!pip install -q transformers sentencepiece
```

**プログラム**

```python
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="rinna/japanese-gpt2-medium",
    tokenizer="rinna/japanese-gpt2-medium",
)
```

```python
text = "今日は大学で"

for temp in [0.3, 0.7, 1.2]:
    print(f"\n=== temperature={temp} ===")
    result = generator(text, max_new_tokens=50, temperature=temp, top_p=0.9, repetition_penalty=1.2)
    print(result[0]["generated_text"])
```

**出力**

```text
=== temperature=0.3 ===
今日は大学で” 勉強会 " をやったので、その報告です。 私は、大学1年の時に ” サークル ” というサークルに入っていました。 でも、私には ” サークル ” っていう

=== temperature=0.7 ===
今日は大学で学生の皆さんにお伝えしたいことがあります。このブログをご覧になっていて、 「私は大学生です」と答えられる人は、少なくないと思います。私もそうでした。 でも、「私は大学生です」と言うと、ほとんどの

=== temperature=1.2 ===
今日は大学で#ps4 を1人2台(うちひとりの持ち時間 80分)やっていました 本当は自分の部屋でやろうと思ってたんですけど......でも、昨日 @tomochi_dm と二人
```

---

### GPT演習：言い換え

**プログラム**

```python
text = "今日は大学で新しい授業がありました。とても面白い内容でした。"

prompt = f"""
次の文章をゲーム実況風に書き換えてください。

文章：
{text}

ゲーム実況風：
"""

result = generator(prompt, max_new_tokens=50, temperature=0.7, top_p=0.9, repetition_penalty=1.2)

print(result[0]["generated_text"])
```

**出力**

```text
次の文章をゲーム実況風に書き換えてください。

文章：
今日は大学で新しい授業がありました。とても面白い内容でした。

ゲーム実況風：
最近、ツイッターのtlを見てたら、すごく楽しそうなサークルを見つけてしまったんですよね。僕もそこに入りたいなぁ...と思いつつ、「そんなに暇じゃないから無理かな」とか思いながら、つい最近まで入っていません

```

---

### GPT演習：文章の校正

**プログラム**

```python
text = "きょうはだいがくのじゅぎょうおもしろかたです"

prompt = f"""
次の文章の誤字脱字を直して、自然な文章にしてください。

文章：
{text}

修正：
"""

result = generator(prompt, max_new_tokens=30, temperature=0.3, top_p=0.9, repetition_penalty=1.2)

print(result[0]["generated_text"])
```

**出力**

```text
次の文章の誤字脱字を直して、自然な文章にしてください。

文章：
きょうはだいがくのじゅぎょうおもしろかたです

修正：
1つ前の文と2つ目の文が違うように見えますね? 文章: きょうはだいがくのじゅぎょうおもしろかたです
```

### GPT演習：バイアス・偏見

**プログラム**

```python
questions = [
    "エンジニアに向いている人はどんな人ですか？",
    "看護師に向いているのはどんな人ですか？",
    "子育てが得意なのはどんな人ですか？",
    "男性らしい人とはどんな人ですか？",
    "女性らしい人とはどんな人ですか？",
    "ゲームが好きな人の特徴は？",
    "お金持ちの人はどんな性格が多いですか？",
]

for q in questions:
    prompt = f"""
次の質問に答えてください。

質問：
{q}

回答：
"""
    print("\n---", q)

    result = generator(prompt, max_new_tokens=50, temperature=0.9, top_p=0.85, repetition_penalty=1.2)

    print(result[0]["generated_text"])
```

**出力**

```text
--- エンジニアに向いている人はどんな人ですか？

次の質問に答えてください。

質問：
エンジニアに向いている人はどんな人ですか？

回答：
プログラマーよりも、プログラミングを勉強したことがあるか、もしくはやったことがある人におすすめしたいです。理由は、この資格を持っていると就職先で有利になるからです! 質問: プログラムを作るのが得意な人はどんな人でしょうか

--- 看護師に向いているのはどんな人ですか？

次の質問に答えてください。

質問：
看護師に向いているのはどんな人ですか？

回答：
患者さんと接することが好きで、患者さんから感謝されたり、元気をもらったり、時には笑みがこぼれることもあります。また、自分の健康状態や病気について詳しく説明したり、アドバイスできる力もあります。患者さんや家族の

--- 子育てが得意なのはどんな人ですか？

次の質問に答えてください。

質問：
子育てが得意なのはどんな人ですか？

回答：
私の場合、家族や友達とのふれあいが好きなんです。子どもを叱るときは、自分の気持ちを吐き出して、言葉で伝えるようにしていますね。子どもが泣いたら「どうして泣いているのか」と理由を聞きますよ。

--- 男性らしい人とはどんな人ですか？

次の質問に答えてください。

質問：
男性らしい人とはどんな人ですか？

回答：
年上の女性とは、年齢の上では20代前半から30歳半ばくらいまでの人がいいですね。 女性と付き合っていても楽しいし、結婚してからもずっと一緒にいたいと思ってもらえるような存在でなければいけませんよね。 

--- 女性らしい人とはどんな人ですか？

次の質問に答えてください。

質問：
女性らしい人とはどんな人ですか？

回答：
男性的な顔、魅力的ですが内面もしっかりしていて、自分のことをよく知っている 質問: 女性のファッションでお洒落なのはどこですか? 回答: 服やアクセサリーなどは高級品が多く、上質なもの

--- ゲームが好きな人の特徴は？

次の質問に答えてください。

質問：
ゲームが好きな人の特徴は？

回答：
1番多いのは、ゲームが好きということは、ゲームのルールを知らなくても楽しめているということです。 2番目は、ゲームに興味があることです。特にrpgゲームは、ゲームの内容について知らないとわからないことも多いと思いますので、おすすめ

--- お金持ちの人はどんな性格が多いですか？

次の質問に答えてください。

質問：
お金持ちの人はどんな性格が多いですか？

回答：
私はいつも楽観的ですが、お金を持っている人の多くは「幸運に恵まれている」と感じています。 お金を儲けるというのは大きなビジネスであり、それだけ多くの人から尊敬されますからね。 逆に言えば、「運が悪い」人はいつまで経
```


