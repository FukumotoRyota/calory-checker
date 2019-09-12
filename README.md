# 手順

## 各ファイルの説明

### ml/learning_ex.py

学習してモデルを作る例。

### ml/using_ex.py

モデルをロードして使う例。

### mymodule.py

ここに色々な関数が入っている。各自色々作る。

## 1. 画像収集

以下のコマンドで画像を集める

```
googleimagesdownload --keywords "リンゴ"
```

[s3](https://s3.console.aws.amazon.com/s3/buckets/foodimages7458/?region=ap-northeast-1&tab=overview)にアップロード

## 2. 画像編集

以下のコマンドでs3から画像を落としてくる。

```
aws s3 cp s3://foodimages7458 . --recursive
```

関係ない画像や適していない画像を除外する。例えば、対象物が端に写っているものは除外すべき。

## 3. 学習

`learning_ex.py`が例。適当にコードを変えて実行する。testデータはこの中で使用する。モデルはmodel.h5というファイルで生成される。

## 4. モデルの使用

`using_ex.py`が例。適当にコードを変えて実行する。
