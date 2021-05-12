#ヒストグラムを作成するPythonコード(gazou4-1.py)

# coding: utf-8
# 一様乱数、算術計算、グラフ描画に必要なライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt
 
# 乱数を足し合わせる回数
N = 1200
# 生成する乱数の個数を指定
n = 10000
f = []
s = 0
 
# 平均値0にするために、平均値N/2を差し引くことに留意する。
for i in range(N):
    s += np.random.rand(n)
f.append(s - N/2)
 
# 平均値と標準偏差の表示
print("Average : " + str(np.average(f)))
print("std-div : " + str(np.std(f)))
 
# スタージェスの公式より階級幅pを14、または平方根の100とする。
p = 100
plt.hist(f, bins=p)
plt.ylabel("Frequency")
plt.show()

















#乱数の個数nを変化させ標準偏差と平均値の変化を散布図としてプロットするPythonコード(gazou4-2.py)

# coding: utf-8
# 一様乱数、算術計算、グラフ描画に必要なライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt
 
# 初期化など
ave_list = []
std_list = []
# 0からnまでの乱数の個数を生成する。このnを変えると乱数の個数の最大値が変わる。
n = 1000
 
 
def returner(n):
    # 乱数を足し合わせる回数
    N = 1200
    f = []
    s = 0
    for i in range(N):
        s = s + np.random.rand(n)
    f.append(s - N / 2)
    return np.average(f), np.std(f)
 
 
# 乱数の個数nを変化させてリストに追加する。
for i in range(n):
    x = i + 1
    # 100エポックごとに進捗を表示させたい場合は以下を表示する
    if x % 100 == 0:
        print("We are at:" + str(x))
    # リストに順次乱数の個数nに対する平均値と標準偏差を追加する。
    # returner()関数はタプルで平均と標準偏差を順に返すためaveとstdでそれぞれ要素を指定する。
    ave_list.append(returner(x)[0])
    std_list.append(returner(x)[1])
 
# 乱数の個数をプロットするためのリスト
X = []
for i in range(n):
    x = i
    X.append(x)
 
# 標準偏差の散布図の描画
plt.scatter(X, ave_list, s=10)
plt.xlabel("Number of random numbers")
plt.ylabel("Standard deviation")
plt.grid(True)
plt.show()
# 平均値の散布図の描画
plt.scatter(X, std_list, s=10)
plt.xlabel("Number of random numbers")
plt.ylabel("Average")
plt.grid(True)
plt.show()
