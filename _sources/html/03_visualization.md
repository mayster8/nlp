# 可視化

```{admonition} 今回のゴール
- 単語頻度の棒グラフを作成する
- ワードクラウドを作成する
- 共起ネットワークの作成と、その他の可視化の用途を知る
```

---

## 頻出単語の棒グラフ
テキストデータを解析した後は、**頻出する単語を棒グラフで可視化**することで直感的に理解できる。  
数値だけでなく、どの単語が多いのかをグラフで示すと一目で把握できる。  

### プログラム例（頻出単語の棒グラフ）

**ライブラリをインストール（必要に応じて）**
```python
!pip -q install spacy ginza ja-ginza
```

**グラフの日本語表示のライブラリをインストール**
```python
!pip -q install japanize_matplotlib
```

**プログラム**
```python
import spacy
import matplotlib.pyplot as plt
import japanize_matplotlib
from collections import Counter

nlp = spacy.load("ja_ginza")

# 対象文書
text = """
昨日の夜は東京の有名なラーメン店に行きました。
そのラーメン店はスープが濃厚で、ラーメンの麺はとても美味しかったです。
私はラーメンと餃子を注文し、友達はラーメンとチャーハンを注文しました。
店内は観光客でにぎわっていて、東京でも人気のラーメン店だと感じました。
最後にアイスを食べましたが、やはりラーメンが一番印象的でした。
"""

# 名詞・固有名詞・形容詞
target_pos = ("NOUN", "PROPN", "ADJ")

# ストップワード
stop_words = ["\n", " ", "　"]

# 単語リストを作成
doc = nlp(text)
all_terms = []
for token in doc:
    if token.pos_ in target_pos and token.lemma_ not in stop_words:
        all_terms.append(token.lemma_)

# 頻度を集計
counter = Counter(all_terms)
common = counter.most_common(10)

# データをプロット用に分解
words = []
counts = []
i = 0
while i < len(common):
    item = common[i]
    words.append(item[0])
    counts.append(item[1])
    i = i + 1

# 棒グラフ表示
plt.bar(words, counts)
plt.title("頻出単語の上位10語")
plt.xlabel("単語")
plt.ylabel("出現回数")
plt.show()
```

**出力**

<img src=../static/out-basic_analysis-bar.png>

---

## ワードクラウド
**ワードクラウド**は、単語の出現頻度に応じて文字の大きさを変えて表示する手法。  
単語の重要度を直感的に把握するのに適している。  

### プログラム例（ワードクラウド）

**Google Colab（Linux環境）上で日本語フォントをインストール**
```python
!apt-get -y install fonts-ipaexfont
```

**プログラム**
```python
import spacy
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
from matplotlib import font_manager

nlp = spacy.load("ja_ginza")

# 対象文書
text = """
昨日の夜は東京の有名なラーメン店に行きました。
そのラーメン店はスープが濃厚で、ラーメンの麺はとても美味しかったです。
私はラーメンと餃子を注文し、友達はラーメンとチャーハンを注文しました。
店内は観光客でにぎわっていて、東京でも人気のラーメン店だと感じました。
最後にアイスを食べましたが、やはりラーメンが一番印象的でした。
"""

# 名詞・固有名詞・形容詞
target_pos = ("NOUN", "PROPN", "ADJ")

# ストップワード
stop_words = ["\n", " ", "　"]

# 改行文字（"\n","\r\n"等）を一括で除去。ワードクラウドでは改行文字はエラーとなるため
text = "".join(text.splitlines())

# 単語リストを作成
doc = nlp(text)
all_terms = []
for token in doc:
    if token.pos_ in target_pos and token.lemma_ not in stop_words:
        all_terms.append(token.lemma_)

# 頻度を集計
counter = Counter(all_terms)

freq_dict = {}
for term, cnt in counter.items():
    freq_dict[term] = cnt

jp_font_path = "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"

# WordCloud の生成
wc = WordCloud(
    width=800,
    height=400,
    background_color="white",
    font_path=jp_font_path,
    # max_words=100,
    random_state=0
)

wc = wc.generate_from_frequencies(freq_dict)

# 表示
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.show()
```

**出力**

<img src=../static/out-basic_analysis-wc.png>

---

### 見た目を調整する主なパラメータ  

| パラメータ名                           | 例                                               | 説明                                         |
| -------------------------------- | ----------------------------------------------- | ------------------------------------------ |
| `width`, `height`                | `width=800, height=400`                         | 出力画像のサイズ。横長や縦長を調整。                         |
| `background_color`               | `"white"`, `"black"`                            | 背景色を変更。 `"white"` が見やすい。                   |
| `colormap`                       | `"viridis"`, `"plasma"`, `"tab10"`, `"rainbow"` | 単語のカラーマップを変更。matplotlibのカラーマップ名を指定。        |
| `max_words`                      | `max_words=100`                                 | 表示する単語の最大数。多すぎると読みづらい。                     |
| `min_font_size`                  | `min_font_size=10`                              | 小さい文字の最小サイズを設定。                            |
| `max_font_size`                  | `max_font_size=100`                             | 最大フォントサイズを明示的に指定可能。                        |
| `font_path`                      | `"ipaexg.ttf"`                                  | 日本語フォントのパス（必須）。                            |
| `relative_scaling`               | `0.5`                                           | 頻度と文字サイズの比例度。`0`で均一、`1`で完全比例。              |
| `scale`                          | `2`                                             | 高解像度出力。2倍にすると滑らかに。                         |
| `margin`                         | `2`                                             | 単語間の余白。小さすぎると重なりやすい。                       |
| `prefer_horizontal`              | `0.9`                                           | 横書き単語の割合。`0`に近いと縦文字が増える。                   |
| `collocations`                   | `False`                                         | 「ラーメン 店」のような複合語を結合するかどうか。日本語では `False` 推奨。 |
| `random_state`                   | `42`                                            | 配置の再現性を確保。                                 |
| `contour_width`, `contour_color` | `contour_width=2, contour_color='steelblue'`    | 単語の輪郭線を追加。立体感を出すときに有効。                     |

---



## 共起ネットワーク

- **共起** とは、「同じ文や段落など、一定の範囲内で一緒に出現する単語」のこと。  
- 例えば、「東京」「ラーメン」が同じ文に現れるとき、「東京」と「ラーメン」には共起関係があるといえる。  
- **共起ネットワーク**は、単語をノード、共起関係をエッジ（線）で表したグラフ構造。

---

### 共起ネットワークの特徴

| 観点 | 内容 |
|------|------|
| **目的** | 単語間の関係構造を可視化し、文脈的なまとまりを発見する |
| **入力** | テキスト中の単語と、それらの共起回数 |
| **出力** | ノード（単語）とエッジ（共起関係）から成るネットワーク図 |
| **分析効果** | どの単語が中心的か（中心性）、どんな語群が結びついているか（クラスタ構造） |

---

**共起行列の作成手順**  

1. テキストを文単位に分割する  
2. 各文中に出現する単語の組み合わせを抽出する  
3. 単語ペアごとに共起回数をカウントする  
4. 共起行列（共起頻度表）を作る  

**例**  

> 文1：<u>東京</u>の<u>ラーメン</u>は<u>美味しい</u>。  
> 文2：<u>大阪</u>の<u>ラーメン</u>も<u>人気</u>がある。

- 文1に含まれる単語：`[東京, ラーメン, 美味しい]`  → 共起ペア：(東京, ラーメン)、(東京, 美味しい)、(ラーメン, 美味しい)  
- 文2に含まれる単語：`[大阪, ラーメン, 人気]`  → 共起ペア：(大阪, ラーメン)、(大阪, 人気)、(ラーメン, 人気)

| 単語1 | 単語2 | 共起回数 |
|--------|--------|----------|
| 東京 | ラーメン | 1 |
| 東京 | 美味しい | 1 |
| ラーメン | 美味しい | 1 |
| 大阪 | ラーメン | 1 |
| 大阪 | 人気 | 1 |
| ラーメン | 人気 | 1 |

この表が「共起行列」の一部に相当し、これを使ってネットワークを作ると、
「東京」「大阪」といった地域語を中心に「ラーメン」「人気」「美味しい」といった関連語がつながる構造が得られる。  

<img src=../static/out-conetwork.png>

---

### プログラム例（共起ペアの計算）

**ライブラリインストール（必要に応じて）**  
```python
!pip -q install spacy ginza ja-ginza
```

**プログラム**  
```python
import spacy
import pandas as pd
from collections import Counter
from itertools import combinations

nlp = spacy.load("ja_ginza")

texts = [
    "昨日の夜は東京のラーメン店に行きました。",
    "そのラーメン店はスープが濃厚で、とても美味しかったです。",
    "友達もラーメンが好きで、よく東京で食べます。"
]

target_pos = ("NOUN", "PROPN", "ADJ")
stop_words = ["\n", " ", "　"]

# 共起ペア
pairs = []

# 共起語ペアの抽出とカウント
for text in texts:
    doc = nlp(text)
    words = []
    for token in doc:
        if token.pos_ in target_pos and token.lemma_ not in stop_words:
            words.append(token.lemma_)
    for w1, w2 in combinations(set(words), 2):
        pairs.append(tuple(sorted([w1, w2])))

co = Counter(pairs)

# 共起回数をカウント
co = Counter(pairs)

# DataFrame（単語ペア・共起回数）
df_co = pd.DataFrame(co.items(), columns=["単語ペア", "共起回数"])
df_co
```

**出力**
| |単語ペア|共起回数|
|---|---|---|
|0|ラーメン,夜|1|
|1|ラーメン,東京|2|
|2|ラーメン,店|2|
|3|ラーメン,昨日|1|
|4|夜,東京|1|
|5|夜,店|1|
|6|夜,昨日|1|
|7|店,東京|1|
|8|昨日,東京|1|
|9|店,昨日|1|
|10|スープ,ラーメン|1|
|11|ラーメン,濃厚|1|
|12|ラーメン,美味しい|1|
|13|スープ,店|1|
|14|スープ,濃厚|1|
|15|スープ,美味しい|1|
|16|店,濃厚|1|
|17|店,美味しい|1|
|18|濃厚,美味しい|1|
|19|友達,東京|1|
|20|好き,東京|1|
|21|ラーメン,友達|1|
|22|友達,好き|1|
|23|ラーメン,好き|1|

---

### 共起ネットワークの可視化

**ネットワーク可視化ライブラリをインストール**  
```python
!pip -q install pyvis
```

**プログラム**
```python
import spacy
import pandas as pd
from itertools import combinations
from collections import Counter
from pyvis.network import Network
import networkx as nx
from IPython.display import HTML
from google.colab import files

nlp = spacy.load("ja_ginza")

texts = [
    "昨日の夜は東京のラーメン店に行きました。",
    "そのラーメン店はスープが濃厚で、とても美味しかったです。",
    "友達もラーメンが好きで、よく東京で食べます。"
]


target_pos = ("NOUN", "PROPN", "ADJ")

stop_words = ["\n", " ", "　"]

# 共起ペア抽出
pairs = []
node_freq = Counter()

for text in texts:
    doc = nlp(text)
    words = []
    for token in doc:
        if token.pos_ in target_pos and token.lemma_ not in stop_words:
            words.append(token.lemma_)

    node_freq.update(words)
    for w1, w2 in combinations(set(words), 2):
        pairs.append(tuple(sorted([w1, w2])))

# 共起回数をカウント
co = Counter(pairs)

# DataFrame（単語ペア・共起回数）
df_co = pd.DataFrame(co.items(), columns=["単語ペア", "共起回数"])
# print(df_co)

# ノードの配置を計算（seed を固定すると毎回同じ配置になる）
G = nx.Graph()
for node in node_freq:
    G.add_node(node)
for (a, b), freq in co.items():
    G.add_edge(a, b, weight=freq)

pos = nx.spring_layout(G, seed=1, k=0.5)   # k を大きくするとノードの間隔が広がる

# 共起ネットワーク構築
net = Network(
    height="600px",
    width="100%",
    bgcolor="white",
    font_color="black",
    cdn_resources="in_line"   # Colabでの表示の場合
)

# ノード追加（出現頻度に応じてサイズ変更、位置は計算した座標）
# scale：配置の広がり。ノードが多くて重なる場合は 1000 などに大きくする
scale = 300
for node, freq in node_freq.items():
    x, y = pos[node]
    net.add_node(node, label=node, size=10 + freq * 5, color="#89CFF0",
                 x=float(x) * scale, y=float(y) * scale)

# エッジ追加（共起頻度で太さと濃さを調整）
for (a, b), freq in co.items():

    # 出現頻度が1回以上
    if freq >= 1:
        color = f"rgba(0,0,255,{min(0.3 + 0.2*freq, 1)})"
        net.add_edge(a, b, value=freq, title=f"共起数: {freq}", color=color)

# 物理シミュレーションを止める（ネットワークが静止する）
net.toggle_physics(False)

# HTMLファイルを保存したい場合
# net.write_html("co_network.html")
# files.download("co_network.html")

# Colabで可視化
net.prep_notebook()
html = net.generate_html()
display(HTML(html))

```

**出力**

<img src=../static/out-conetwork2.png>

---

### 共起ネットワークの分析の視点

共起ネットワークを作成したあとは、**グラフの形そのものを読み解く分析**が重要になる。  
ノードの配置やエッジのつながりから、どんな語が中心的な意味を持つのか、どのような話題のまとまりが存在するのかを解釈する。

<!-- 
| 観点 | 説明 |
|------|------|
| **中心語の抽出** | ノードの「次数（接続数）」や「中心性」などを計算し、他の語と多く結びついている中心的な語を特定します。例：ラーメンを中心に「東京」「店」などが広がる。 |
| **クラスタ構造の発見** | ネットワークの中で密につながる語群を探すことで、文脈的に関連するトピックや話題のまとまりを見つける。|
| **関係の方向性と強さ** | エッジの太さ（共起頻度）を比較し、どの関係が強いかを判断する。強い線は共通して登場する語の組み合わせを示す。 |
| **文脈の解釈** | 近接して配置されているノードほど意味的・文脈的に関連性が高いと考えられる。ノード間の距離から、語同士の関連の強弱を推測できる。 |
| **変化の比較** | 時期別・カテゴリ別に共起ネットワークを作ると、話題の変化や注目語の移り変わりを視覚的に追跡できる。 | -->

#### 読み取りのポイント
- **中心語を見る**：ネットワークの中央に位置する語は、そのテーマの「ハブ」になっている。  
- **グループ（クラスタ）を探す**：近い位置に固まる単語群は、同じ文脈や話題を共有している。  
- **線の太さに注目**：共起回数が多い語ほど、強い関係性を持つ。  
- **孤立語も重要**：あまり共起しない語は、特定の文脈でのみ登場する特徴的な語である場合がある。  

共起ネットワークは単なる単語の関係図ではなく、話題構造や意味的関連を読み解く道具として活用できる。


---
<!-- 
### ワードクラウド・共起ネットワーク・クラスタリングの比較

これら3つの手法は、いずれもテキストデータを「見える化」する方法ですが、焦点の当て方と得られる知見が異なる。

| 手法 | 分析の単位 | 主な目的 | 表現形式 | 得られる情報 | 適した用途 |
|------|-------------|-----------|------------|----------------|--------------|
| **ワードクラウド** | 単語 | 頻出単語を視覚的に把握 | 文字の大きさ | 出現頻度が高い単語 | 話題の概要を直感的に示したい場合（例：レビュー全体の印象） |
| **共起ネットワーク** | 単語間のペア | 単語同士の関係構造を理解 | ノードと線（ネットワーク） | 文脈的な関連語・語群 | 話題の構造や関連テーマを分析したい場合（例：SNS上の話題関係） |
| **クラスタリング** | 文書 | 文書群のグループ構造を発見 | グループ化・デンドログラム | 内容が似た文書のまとまり | テキスト集合を分類したい場合（例：レビューや記事の自動分類） |

--- -->
<!-- 
#### ワードクラウド（Word Cloud）
- **目的**：どんな単語が多く出現しているかを一目で把握する。  
- **特徴**：単語の出現頻度が大きさで表現され、直感的・印象的に伝わる。  
- **長所**：可視化が簡単で、授業・報告資料・ダッシュボード向け。  
- **短所**：文脈や関係性がわからない。  
- **使用例**：
  - 商品レビューで「顧客がよく使う言葉」を示す。
  - SNS投稿の中で注目されているキーワードを可視化。

---

#### 共起ネットワーク（Co-occurrence Network）
- **目的**：単語同士の関係（共起）を分析し、語のつながりを構造として捉える。  
- **特徴**：ノードが単語、エッジが共起関係を表すネットワーク構造。  
- **長所**：文脈的なつながりや話題のまとまりを理解できる。  
- **短所**：データ量が多いと見づらくなりやすい。  
- **使用例**：
  - 「どの単語が中心的なテーマと関連しているか」を可視化。
  - SNSやニュース記事で「話題の関係構造」や「言葉のネットワーク」を分析。

---

#### クラスタリング（Clustering）
- **目的**：複数の文書を内容の類似性に基づいて自動的にグループ化する。  
- **特徴**：各文書がベクトル化（TF-IDFなど）され、類似度に基づいて分類される。  
- **長所**：大量の文書から潜在的なトピックやテーマを発見できる。  
- **短所**：結果の解釈に時間がかかる（どんなトピックかを確認する必要あり）。  
- **使用例**：
  - レビューを「味に関する意見」「価格に関する意見」などに分類。  
  - ニュース記事を「政治」「経済」「スポーツ」などに自動的に分ける。  

--- -->

### 可視化手法の使い分けのイメージ

| 分析の目的 | 適した手法 | 補足 |
|-------------|-------------|------|
| 文章の中でどんな単語が多いかを見たい | ワードクラウド | テキスト全体の印象を素早くつかむ |
| 単語がどのようにつながっているかを知りたい | 共起ネットワーク | テーマ間の関係性や共通語を把握 |
| 文書を自動的にグループ分けしたい | クラスタリング | 潜在的なトピック構造を発見する |

---
<!-- 
## まとめ

- **ワードクラウド**：テキスト全体の**概要を俯瞰**する。  
- **共起ネットワーク**：単語間の**関係構造を理解**する。  
- **クラスタリング**：文書全体の**グループ構造を抽出**する。  

これらを組み合わせることで、  
「頻出語（ワードクラウド）→関連語（共起ネットワーク）→文書のまとまり（クラスタリング）」  
という段階的なテキストマイニング分析が可能になる。
 -->
