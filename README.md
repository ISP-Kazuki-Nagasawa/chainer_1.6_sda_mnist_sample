# Chainer Stacked Auto Encoder MNIST Sample
Chainer で MNIST (手書き文字認識) データの次元圧縮を題材にして、Stacked Auto-Encoder を実行するサンプルです。  
  
当社技術ブログ「技ラボ」にて概要を公開しています。  
( http://wazalabo.com/chainer-stacked-auto-encoder.html )

Chainer 1.6 以上 (Chainer 1.6 or 1.7) での動作確認済みです。

Chainer 1.5 未満の場合は、以前のものを使用してください。
( https://github.com/ISP-Kazuki-Nagasawa/chainer_sda_mnist_sample )

## 動作確認済み環境
- Python 2.7 or 3.4
- Chainer 1.6.0 or 1.7.1


## Requeirements
本コードは以下を使用しています。
- chainer
- docopt
- numpy
- six
  
pip コマンドと requirements.txt から簡単にインストールを行うことが出来ます。

        $ sudo pip install -r requirements.txt


## データ
本ソースコードは MNIST データを題材としています。  
MNIST データローダは Chainer のサンプルソースコードを利用しています。  
( https://github.com/pfnet/chainer/blob/master/examples/mnist/data.py )


## 使用法
### 設定
設定は settings.py で行います。以下が設定可能です。
- 表示ログレベル
- 出力ファイル名
- 入力層、中間層のサイズ
- Pretraining 設定
    - batch 数
    - epoch 数
- Finetuning 設定
    - batch 数
    - epoch 数
- 結果出力設定
    - batch 数

### 実行
execute.py にて実行します。

        $ python execute.py <output csv file>

GPU が入っているPCでの実行の場合、GPU ID を設定すると高速に処理できます。

        $ python execute.py -g <GPU ID> <output csv file>


### その他
- Linear Linkのコピーについて  
本プログラムでは、pretraining、fine tuning、結果出力の際、それぞれ別の Chain を利用しています。  
Linear Linkはその度に移し替えているのですが、その際、Chainer の『1つのLinkを複数のChainで使い回すことができない』問題により、各処理前に Link をコピーして使用しています。
Chainer が対応し次第、直したいところです。
    - https://github.com/pfnet/chainer/issues/715





