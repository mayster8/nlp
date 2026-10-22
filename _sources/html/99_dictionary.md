# 辞書の拡張

```{admonition} 今回のゴール
- GiNZAによる固有名詞認識（NER）の仕組みを理解する
- 固有名詞を拡充する方法を知る（NEologd, Wikipedia, Pageviewsなど）
- 拡張辞書を使って固有名詞を1語扱いで抽出できるようにする
```

---

## 固有名詞の拡充

- 固有名詞とは、人名「山田花子」、地名「世田谷区」、組織名「東京都市大学」、商品名「カップヌードル」、作品名「鋼の錬金術師」のように特定の名前を指す用語。
- GiNZAの日本語モデルでは新語や作品名が検出されにくい。
- 例）「カップヌードル」「鋼の錬金術師」などが1語扱いされない。

### 確認： GiNZAによる形態素解析

**ライブラリインストール（必要に応じて）**  
```
!pip -q install spacy ginza ja-ginza
```

**プログラム**
```python
import spacy
nlp = spacy.load("ja_ginza")

texts = [
    "カップヌードルを食べて鋼の錬金術師を見た",
]

# 形態素解析して用語を出力
for text in texts:
  doc = nlp(text)
  tokens = []

  for token in doc:
      tokens.append(f"{token.text}({token.pos_})")

  print(tokens)
```

**出力**
```
['カップ(NOUN)', 'ヌードル(NOUN)', 'を(ADP)', '食べ(VERB)', 'て(SCONJ)', '鋼(NOUN)', 'の(ADP)', '錬金(NOUN)', '術師(NOUN)', 'を(ADP)', '見(VERB)', 'た(AUX)']
```

---

### 固有名詞の拡充の方法

**A）自前で用語を追加した辞書を用いる方法** 
- 抽出したい用語を辞書に登録して形態素解析する  
- 設定が容易だが、どんな用語を登録するか知識が必要  

**B）既存の辞書を活用する方法**  
- MeCab用のNEologd辞書を使う  
- 辞書の更新が容易だが、設定が複雑（特に Google Colab では複雑）  

#### プログラム例（自前で用語を追加する）

**プログラム**
```python
import spacy
from spacy.pipeline import EntityRuler

# ==== ２つの形態素解析器を用意 ====
nlp = spacy.load("ja_ginza")      # 通常の形態素解析器
nlp_dic = spacy.load("ja_ginza")  # 辞書拡張した形態素解析器

texts = [
  "カップヌードルを食べて鋼の錬金術師を見た",
]

# ==== 通常の形態素解析 ====
for text in texts:
  doc = nlp(text)
  tokens = []
  for token in doc:
      tokens.append(f"{token.text}({token.pos_})")
      # tokens.append(token.text)
  print(tokens)
print("")

# ==== 辞書を使った形態素解析 ====
# 辞書に用語を追加 
# entity_ruler をパイプラインの先頭に追加する
ruler = nlp_dic.add_pipe("entity_ruler", first=True)
patterns = [
    {"label": "USER_DICT", "pattern": "カップヌードル"},
    {"label": "USER_DICT", "pattern": "鋼の錬金術師"},
]
ruler.add_patterns(patterns)

for text in texts:

  # 形態素解析器の結果に固有名詞として登録
  doc = nlp_dic(text)
  with doc.retokenize() as retokenizer:
      for ent in doc.ents:
          attrs = {"POS": "PROPN"}
          retokenizer.merge(ent, attrs=attrs)

  tokens = []
  for token in doc:
      tokens.append(f"{token.text}({token.pos_})")
      # tokens.append(token.text)
  print(tokens)

```

**出力**
```
['カップ(NOUN)', 'ヌードル(NOUN)', 'を(ADP)', '食べ(VERB)', 'て(SCONJ)', '鋼(NOUN)', 'の(ADP)', '錬金(NOUN)', '術師(NOUN)', 'を(ADP)', '見(VERB)', 'た(AUX)']

['カップヌードル(PROPN)', 'を(ADP)', '食べ(VERB)', 'て(SCONJ)', '鋼の錬金術師(PROPN)', 'を(ADP)', '見(VERB)', 'た(AUX)']
```


---


#### プログラム例（Wikipediaページタイトルの用語追加）

**辞書ファイル**  
- <a href="http://ogawa.isslab.org/nlp/data/wiki_20261014_top_1000.json" download>wiki_20261014_top_1000.json</a>
- <a href="http://ogawa.isslab.org/nlp/data/wiki_20261014_top_10000.json" download>wiki_20261014_top_10000.json</a>
- <a href="http://ogawa.isslab.org/nlp/data/wiki_20261014_top_30000.json" download>wiki_20261014_top_30000.json</a>

上記のファイルをダウンロードして、Google Colab に保存しておく

**プログラム**
```python
import spacy, json
from spacy.pipeline import EntityRuler

# ==== ２つの形態素解析器を用意 ====
nlp = spacy.load("ja_ginza")      # 通常の形態素解析器
nlp_dic = spacy.load("ja_ginza")  # 辞書用の形態素解析器

# ==== 辞書に用語を追加 ====
dic = "wiki_20261014_top_1000.json"
# dic = "wiki_20261014_top_10000.json"
# dic = "wiki_20261014_top_30000.json"

with open(dic, "rt", encoding="utf-8") as f:
    patterns = json.load(f)

# entity_ruler をパイプラインの先頭に追加する
# （先頭に置くと、用語の登録時に分かち書きだけが実行されるので速くなる）
ruler = nlp_dic.add_pipe("entity_ruler", first=True)
ruler.add_patterns(patterns)
```

```python
texts = [
"カップヌードルを食べて鬼滅の刃を見た",
]


# == 通常の形態素解析 ==
for text in texts:
  doc = nlp(text)
  tokens = []
  for token in doc:
      tokens.append(f"{token.text}({token.pos_})")
      # tokens.append(token.text)
  print(tokens)
print("")

# ==== 辞書を使った形態素解析 ====
# 辞書に用語を追加 
for text in texts:

  # 形態素解析器の結果に固有名詞として登録
  doc = nlp_dic(text)
  with doc.retokenize() as retokenizer:
      for ent in doc.ents:
          attrs = {"POS": "PROPN"}
          retokenizer.merge(ent, attrs=attrs)

  tokens = []
  for token in doc:
      tokens.append(f"{token.text}({token.pos_})")
      # tokens.append(token.text)
  print(tokens)
```

**出力**
```
['カップ(NOUN)', 'ヌードル(NOUN)', 'を(ADP)', '食べ(VERB)', 'て(SCONJ)', '鬼(NOUN)', '滅(PRON)', 'の(ADP)', '刃(NOUN)', 'を(ADP)', '見(VERB)', 'た(AUX)']

['カップヌードル(PROPN)', 'を(ADP)', '食べ(VERB)', 'て(SCONJ)', '鬼滅の刃(PROPN)', 'を(ADP)', '見(VERB)', 'た(AUX)']
```

---

### 参考プログラム（演習用）

#### Wikipediaのページ（概要）を取得してCSVで保存

`keyword` に調べたいWikipediaのページ名を指定する。  
「概要」（または「概観」「概説」）の節があればその節を、なければ冒頭の導入文を取得し、`キーワード.csv`（列名は `text`）で保存する。

**プログラム**
```python
import requests
import pandas as pd
from bs4 import BeautifulSoup

# ==== 調べたいキーワード（Wikipediaのページ名） ====
keyword = "鬼滅の刃"

url = "https://ja.wikipedia.org/wiki/" + keyword

# 「概要」として扱う見出し名
SUMMARY_HEADINGS = ["概要", "概観", "概説"]


def is_heading(el, level=None):
    """
    見出しかどうかを判定する（level=2 なら h2 の大見出しだけを見出しとみなす）
    """
    if el.name is None:
        return False
    if level == 2:
        names = ["h2"]
        classes = ["mw-heading2"]
    else:
        names = ["h2", "h3", "h4", "h5", "h6"]
        classes = ["mw-heading"]
    if el.name in names:
        return True
    # 新しい Wikipedia の HTML では見出しが <div class="mw-heading ..."> で囲まれている
    if el.name == "div":
        for c in el.get("class", []):
            if c in classes:
                return True
    return False


def get_section_text(soup, heading_id):
    """
    指定した見出しの節の段落を取得する（見出しがなければ空文字）
    """
    heading = soup.find(id=heading_id)
    if heading is None:
        return ""
    start = heading.find_parent("div", class_="mw-heading")
    if start is None:
        start = heading
    texts = []
    for el in start.find_next_siblings():
        if is_heading(el, level=2):    # 次の大見出しで終了
            break
        if el.name == "p":
            t = el.get_text(strip=True)
            if t:
                texts.append(t)
    return "\n".join(texts)


def get_lead_text(soup):
    """
    最初の見出しより前にある段落（導入文）を取得する
    """
    # 最初の見出しを探す
    first_heading = None
    for el in soup.find_all(["h2", "div"]):
        if is_heading(el):
            first_heading = el
            break

    if first_heading is not None:
        paragraphs = first_heading.find_all_previous("p")
        paragraphs.reverse()           # find_all_previous は逆順なので並べ直す
    else:
        paragraphs = soup.find_all("p")

    texts = []
    for p in paragraphs:
        if p.find_parent("table") is not None:   # 情報ボックス（表）の中は除く
            continue
        t = p.get_text(strip=True)
        if t:
            texts.append(t)
    return "\n".join(texts)


def get_wikipedia_summary(url):
    """
    WikipediaページURLから「概要」部分のテキストを取得
    （概要の節があればその節、なければ冒頭の導入文）
    """
    # renderモードを付加
    if "action=render" not in url:
        if "?" in url:
            url += "&action=render"
        else:
            url += "?action=render"

    # User-Agent設定
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }

    # ページ取得
    response = requests.get(url, headers=headers)
    response.encoding = response.apparent_encoding

    # BeautifulSoupで解析
    soup = BeautifulSoup(response.text, "html.parser")

    # 脚注の番号（[1] など）を削除
    for sup in soup.find_all("sup", class_="reference"):
        sup.decompose()

    # 1) 概要の節（概要・概観・概説）を探す
    for name in SUMMARY_HEADINGS:
        text = get_section_text(soup, name)
        if text:
            return text

    # 2) なければ冒頭の導入文
    return get_lead_text(soup)


def split_sentences(text):
    """
    文章を1文ずつに分割する（「。」「！」「？」で区切る。かっこの中では区切らない）
    """
    sentences = []
    buf = ""
    depth = 0                      # かっこの深さ
    for ch in text:
        buf += ch
        if ch in "「『（(":
            depth += 1
        elif ch in "」』）)":
            if depth > 0:
                depth -= 1
        elif ch in "。！？!?" and depth == 0:
            sentences.append(buf)
            buf = ""
        elif ch == "\n":           # 段落の切れ目でも区切る
            sentences.append(buf)
            buf = ""
            depth = 0
    sentences.append(buf)

    # 空の文や、句読点だけの文（「。」など）を除く
    results = []
    for s in sentences:
        s = s.strip()
        if s.strip("。、！？!?") != "":
            results.append(s)
    return results


# ==== 実行 ====
text = get_wikipedia_summary(url)

if text:
    sentences = split_sentences(text)

    # 1文1行の表にして保存
    df = pd.DataFrame({"text": sentences})
    filename = keyword + ".csv"
    df.to_csv(filename, index=False, encoding="utf-8-sig")

    print(len(sentences), "文を", filename, "に保存しました")
    display(df)
else:
    print("概要が取得できませんでした。")
```

**出力**（`keyword = "鬼滅の刃"` の場合）

```text
13 文を 鬼滅の刃.csv に保存しました
```

| | text |
|---|---|
| 0 | 大正時代の日本を舞台に、主人公・竈門 炭治郎が鬼と化した妹・竈門 禰󠄀豆子を人間に戻すために鬼たちと戦う姿を描く、和風の剣戟奇譚。 |
| 1 | 単行本の累計発行部数は、2025年7月時点で全世界で2億2000万部を突破しており、そのうち5600万部は海外での発行部数である。 |
| 2 | 第23巻の初週売上は、限定版を含めて国内出版史上最高記録となる343.1万部を記録。 |
| 3 | 2020年オリコン年間コミックランキング作品別の期間内売上で8,234.5万部を記録し、作品別歴代年間最高売上を獲得した。 |
| 4 | また、連載開始から4年と22巻での1億部到達は、それまで『ONE PIECE』が保持していた7年と36巻を大幅に塗り替える「史上最速の1億部突破」である。 |
| 5 | 2019年にはufotable制作でテレビアニメ化され、物語の序章を描く第一期『竈門炭治郎 立志編』が放送された。 |
| 6 | 2020年には物語の中盤を描く劇場アニメ『無限列車編』が公開された。 |
| 7 | 同作品の日本国内での興行収入は404.3億円に達し、それまで『千と千尋の神隠し』が保持していた316億8,000万円を塗り替え、日本歴代興行収入第1位となった。 |
| 8 | 全世界の興行収入は517億円に上り、全世界で観客4135万人を動員し、「世界最高の興行収入の日本アニメ映画」としてギネス世界記録に認定されている。 |
| 9 | 2021年10月から2022年2月まで、テレビアニメ第二期として『無限列車編』の再編集版とその続編となる『遊郭編』が、2023年4月から6月まで、テレビアニメ第三期として『刀鍛冶の里編』が放送された。 |
| 10 | 2024年5月から6月まで、テレビアニメ第四期となる『柱稽古編』が放送された。 |
| 11 | その続編となる劇場アニメ『劇場版「鬼滅の刃」無限城編』が三部作で制作されることが決定し、『第一章 猗窩座襲来』が2025年7月に公開された。 |
| 12 | その他、舞台化やゲーム化、玩具化、企業やテーマパークとのコラボレーションなど、メディアミックスが多様に展開されており、本作の経済規模は1兆円を超えている。 |
