import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from bisect import bisect_left


clusterCount = 0
alpha0, beta0, beta0_alpha0 = 1, 0.5, 0.5
gamma0 = 1
mu0 = 1
totalblock = 0
samplesize = 0
lazy_size = np.zeros(3, dtype=np.int32)
lazy_cumsum = np.zeros(3, dtype=np.float64)
lazy_squaresum = np.zeros(3, dtype=np.float64)
p = np.zeros(3, dtype=np.float64)
lnp = np.zeros(3, dtype=np.float64)


class Info:
    def __init__(self, mu: float = 0.0, sigma: float = 1.0, size: int = 1, idx: int = -1):
        self.mu = mu
        self.sigma = sigma
        self.size = size
        self.idx = idx


class Cluster:
    k = 1

    def __init__(self, cumsum: float, squaresum: float, size: int = 1) -> None:
        # prior: E=alpha/beta; var=alpha/(beta * beta)
        self.inv_var = np.random.gamma(shape=alpha0, scale=1 / beta0)
        # self.var = 1 / self.inv_var
        self.inv_sigma = math.sqrt(self.inv_var)
        # prior: E=mu; var=k * self.var
        self.mu = np.random.normal(
            mu0, 1 / self.inv_sigma / np.sqrt(Cluster.k))
        self.size = size
        self.count = 1
        self.cumsum = cumsum
        self.squaresum = squaresum

    def add(self, cumsum: float, squaresum: float, size: int, block: bool = False) -> None:
        self.size += size
        self.cumsum += cumsum
        self.squaresum += squaresum
        if block:
            self.count += 1
        # if self.squaresum < 0:
        #     print("add<0:", self.squaresum)

    def remove(self, cumsum: float, squaresum: float, size: int, block: bool = False) -> None:
        self.size -= size
        self.cumsum -= cumsum
        self.squaresum -= squaresum
        if block:
            self.count -= 1
        # if self.squaresum < 0:
        #     print("remove<0:", self.squaresum)

    def prob(self, cumsum: float, squaresum: float, size: int) -> float:
        return math.exp((2 * self.mu * cumsum - size * self.mu * self.mu - squaresum) * self.inv_var / 2) \
            * math.pow(self.inv_sigma, size)

    def lnprob(self, cumsum: float, squaresum: float, size: int) -> float:
        return (2 * self.mu * cumsum - size * self.mu * self.mu - squaresum) * self.inv_var / 2 \
            + size * math.log(self.inv_sigma)

    def update(self) -> None:
        mun = (Cluster.k * mu0 + self.cumsum) / (Cluster.k + self.size)
        betan = beta0 + (self.squaresum - self.cumsum *
                         self.cumsum / self.size) / 2
        betan += Cluster.k * (self.cumsum * self.cumsum / self.size + mu0 * mu0 * self.size - 2 * mu0 * self.cumsum) \
            / (2 * (Cluster.k + self.size))
        self.inv_var = np.random.gamma(
            shape=alpha0 + self.size / 2, scale=1 / betan)
        self.inv_sigma = math.sqrt(self.inv_var)
        self.mu = np.random.normal(
            mun, 1 / self.inv_sigma / math.sqrt(Cluster.k + self.size))


clusters = dict[int, Cluster]()


def prob_newcluster(cumsum: float, squaresum: float, size: int) -> float:
    alphan = alpha0 + size / 2
    kn = Cluster.k + size
    betan = beta0 + (squaresum - cumsum * cumsum / size) / 2
    betan += Cluster.k * (cumsum * cumsum / size + mu0 *
                          mu0 * size - 2 * mu0 * cumsum) / (2 * kn)
    temp = math.gamma(alphan) * beta0_alpha0 / gamma0 * \
        math.pow(Cluster.k / kn, 0.5)
    return temp / math.pow(betan, alphan)


def lnprob_newcluster(cumsum: float, squaresum: float, size: int) -> float:
    alphan = alpha0 + size / 2
    kn = Cluster.k + size
    betan = beta0 + (squaresum - cumsum * cumsum / size) / 2
    betan += Cluster.k * (cumsum * cumsum / size + mu0 *
                          mu0 * size - 2 * mu0 * cumsum) / (2 * kn)
    temp1 = math.log(math.gamma(alphan) * beta0_alpha0 / gamma0)
    temp2 = math.log(Cluster.k / kn) / 2
    return temp1 - alphan * math.log(betan) + temp2


def GibbsC(data: np.ndarray, w: np.ndarray, c: np.ndarray, idx2cluster: list[int],
           size: np.ndarray, cumsum: np.ndarray, squaresum: np.ndarray, dp: float) -> None:
    global clusterCount, totalblock
    length = len(data)
    lnw = np.log(w)
    lnweight = np.log(1 - w)
    cumlnw = 0
    for i in range(length - 1, -1, -1):
        lnweight[i] += cumlnw
        cumlnw += lnw[i]
    # 检验lazy延迟操作正确性(1)
    # prev_size = 0
    # prev_cumsum = 0.0
    # prev_squaresum = 0.0
    # for i in range(length):
    #     if c[i] == i:
    #         prev_size += size[i]
    #         prev_cumsum += cumsum[i]
    #         prev_squaresum += squaresum[i]
    for i in range(length - 1, -1, -1):
        cumlnw -= lnw[i]
        lnweight[i] = cumlnw
        size[i] += lazy_size[i]
        cumsum[i] += lazy_cumsum[i]
        squaresum[i] += lazy_squaresum[i]
        if c[i] == i:
            totalblock -= 1
            if clusters[idx2cluster[i]].count == 1:
                clusters.pop(idx2cluster[i])
            else:
                clusters[idx2cluster[i]].remove(
                    cumsum[i], squaresum[i], size[i], True)
                clusters[idx2cluster[i]].update()
        else:
            j = c[i]
            lazy_size[j] += lazy_size[i]
            lazy_cumsum[j] += lazy_cumsum[i]
            lazy_squaresum[j] += lazy_squaresum[i]
            lazy_size[j] -= size[i]
            lazy_cumsum[j] -= cumsum[i]
            lazy_squaresum[j] -= squaresum[i]
            clusters[idx2cluster[i]].remove(cumsum[i], squaresum[i], size[i])
            clusters[idx2cluster[i]].update()
        lazy_size[i] = 0
        lazy_cumsum[i] = 0.0
        lazy_squaresum[i] = 0.0
        idx2cluster[i] = -1
        exlength = i + len(clusters) + 1
        for prevnode in range(i):
            lnweight[prevnode] -= lnw[i]
            lnp[prevnode] = lnweight[prevnode] + \
                clusters[idx2cluster[prevnode]].lnprob(
                    cumsum[i], squaresum[i], size[i])
        # 检验权重和为1
        # checksum1 = np.sum(np.exp(lnweight[: (i + 1)]))
        # if abs(checksum1 - 1) > 1e-6:
        #     print("检验权重和为1: ", checksum1)
        k = 0
        idx2key = dict[int, int]()
        for clusterIdx in clusters:
            cluster = clusters[clusterIdx]
            lnp[i + k] = math.log(cluster.count / (totalblock + dp)) + \
                lnweight[i] + cluster.lnprob(cumsum[i], squaresum[i], size[i])
            idx2key[i + k] = clusterIdx
            k += 1
        lnp[i + k] = math.log(dp / (totalblock + dp)) \
            + lnweight[i] + lnprob_newcluster(cumsum[i], squaresum[i], size[i])
        maxlnp = -1e9
        for j in range(exlength):
            maxlnp = max(maxlnp, lnp[j])
        p[0] = math.exp(lnp[0] - maxlnp)
        for j in range(1, exlength):
            p[j] = math.exp(lnp[j] - maxlnp)
            p[j] += p[j - 1]
        prev = bisect_left(p, np.random.uniform(low=0.0, high=p[exlength - 1]), hi=exlength)
        if prev < i:
            c[i] = prev
            j = c[i]
            lazy_size[j] += size[i]
            lazy_cumsum[j] += cumsum[i]
            lazy_squaresum[j] += squaresum[i]
            idx2cluster[i] = idx2cluster[j]
            clusters[idx2cluster[i]].add(cumsum[i], squaresum[i], size[i])
            clusters[idx2cluster[i]].update()
        elif prev == exlength - 1:
            c[i] = i
            idx2cluster[i] = clusterCount
            temp = Cluster(cumsum[i], squaresum[i], size[i])
            temp.update()
            clusters[clusterCount] = temp
            clusterCount += 1
            totalblock += 1
        else:
            c[i] = i
            idx2cluster[i] = idx2key[prev]
            clusters[idx2cluster[i]].add(
                cumsum[i], squaresum[i], size[i], True)
            clusters[idx2cluster[i]].update()
            totalblock += 1
    for i in range(1, length):
        idx2cluster[i] = idx2cluster[c[i]]
    # 检验连通块数
    # a = 0
    # for b in clusters:
    #     a += clusters[b].count
    # if a != totalblock:
    #     print("检验连通块数量: ", a == totalblock)
    # 检验lazy延迟操作正确性(2)
    # for i in range(length):
    #     if c[i] == i:
    #         prev_size -= size[i]
    #         prev_cumsum -= cumsum[i]
    #         prev_squaresum -= squaresum[i]
    # if prev_size or abs(prev_cumsum) > 1e-6 or prev_squaresum > 1e-6:
    #     print("lazy延迟操作异常")


def GibbsW(w: np.ndarray, c: np.ndarray, an: np.ndarray, bn: np.ndarray) -> None:
    length = len(c)
    A, B = [0] * (length + 1), [0] * length
    for i in range(length - 1, -1, -1):
        if i:
            A[i - 1] += 1
        if c[i] != i:
            A[c[i]] -= 1
            B[c[i]] += 1
        A[i] += A[i + 1]
        w[i] = np.random.beta(an[i] + A[i], bn[i] + B[i])


# 支持cluster方差不共享
def Gibbs(data: list[np.ndarray], an: list[np.ndarray], bn: list[np.ndarray],
          mu_0: float = 0.0, k: float = 0.005, alpha_0: float = 2.0, beta_0: float = 0.0063,
          niterate: int = 100, dp: float = 2.0, c: list[np.ndarray] = None, draw: bool = False,
          title: str = None, path: str = None) -> tuple[list[Info], list[list[int]]]:
    global clusterCount, mu0, alpha0, beta0, gamma0, beta0_alpha0, totalblock, samplesize, p, lnp
    global lazy_size, lazy_cumsum, lazy_squaresum
    mu0 = mu_0
    alpha0, beta0, Cluster.k = alpha_0, beta_0, max(k, 0.001)
    gamma0, beta0_alpha0 = math.gamma(alpha0), beta0 ** alpha0
    ndata = len(data)
    w = [np.random.beta(an[i], bn[i]) for i in range(ndata)]
    size = [np.ones(len(data[i]), dtype=np.int32) for i in range(ndata)]
    cumsum = [data[i].copy() for i in range(ndata)]
    squaresum = [(data[i] * data[i]) for i in range(ndata)]
    max_size = max(len(data[i]) for i in range(ndata))
    lazy_size = np.zeros(max_size, dtype=np.int32)
    lazy_cumsum = np.zeros(max_size, dtype=np.float64)
    lazy_squaresum = np.zeros(max_size, dtype=np.float64)
    num_cluster = []
    if c is None:
        c = [np.array([j for j in range(len(data[i]))], dtype=np.int32)
             for i in range(ndata)]
    elif c[0].__class__ == list:
        c = [np.array(c[i], dtype=np.int32) for i in range(ndata)]
    idx2cluster = [[-1] * len(data[i]) for i in range(ndata)]
    samplesize = 0
    llf_square = llf_sum = 0.0
    ndrop = max(50, int(0.3 * niterate + 0.0001))
    for i in range(ndata):
        samplesize += len(data[i])
        for j in range(len(data[i]) - 1, -1, -1):
            if c[i][j] == j:
                idx2cluster[i][j] = clusterCount
                clusters[clusterCount] = Cluster(
                    cumsum[i][j], squaresum[i][j], size[i][j])
                clusterCount += 1
            else:
                prev = c[i][j]
                size[i][prev] += size[i][j]
                cumsum[i][prev] += cumsum[i][j]
                squaresum[i][prev] += squaresum[i][j]
        for j in range(len(data[i])):
            if c[i][j] != j:
                idx2cluster[i][j] = idx2cluster[i][c[i][j]]
    for i in clusters:
        clusters[i].update()
    totalblock = clusterCount
    mu_sum = [np.zeros(len(data[i]), dtype=np.float64)
              for i in range(len(data))]
    inv_var_sum = [np.zeros(len(data[i]), dtype=np.float64)
                   for i in range(len(data))]
    p = np.zeros(samplesize + max_size + 5, dtype=np.float64)
    lnp = np.zeros(samplesize + max_size + 5, dtype=np.float64)
    for i in range(niterate):
        for j in range(ndata):
            GibbsC(data[j], w[j], c[j], idx2cluster[j],
                   size[j], cumsum[j], squaresum[j], dp)
            GibbsW(w[j], c[j], an[j], bn[j])
        num_cluster.append(len(clusters))
        llf = - 0.5 * samplesize * math.log(2 * math.pi)
        for j in range(ndata):
            for k in range(len(data[j])):
                cluster = clusters[idx2cluster[j][k]]
                temp1 = cluster.mu
                temp2 = cluster.inv_var
                llf += - temp2 / 2 * ((data[j][k] - temp1) ** 2) \
                    + math.log(temp2) / 2
                if i >= ndrop:
                    mu_sum[j][k] += temp1
                    inv_var_sum[j][k] += temp2
        print("{:d} iteration: K={:d}, llf={:.4f}".format(
            i + 1, len(clusters), llf))
    nrest = niterate - ndrop
    dic = 1e9
    # prob = None
    if nrest > 0:
        for i in range(ndata):
            mu_sum[i] /= nrest
            inv_var_sum[i] /= nrest
        dic = 2 * sum([sum([(data[i][j] - mu_sum[i][j]) ** 2 / 2 * inv_var_sum[i][j]
                            - np.log(inv_var_sum[i][j]) / 2
                            for j in range(len(data[i]))])
                       for i in range(ndata)]) \
            + llf_square / nrest - (llf_sum / nrest) ** 2
        # prob = [np.array([np.exp((data[i][j] - clusters[idx2cluster[i][j]].mu) ** 2
        #                          * (clusters[idx2cluster[i][j]].inv_var / 2)) *
        #                   clusters[idx2cluster[i][j]].inv_sigma
        #                   for j in range(len(data[i]))]) for i in range(ndata)]
        print("DIC: {:.4f}".format(dic))
    if draw:
        plt.figure(figsize=(15, 8))
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        if title:
            plt.title("分配过程(" + title + ")  (dp={dp}, k={k}, alpha0={alpha_0}, beta0={beta_0}, DIC={dic:.4f})"
                      .format(dp=dp, k=Cluster.k, alpha_0=alpha_0, beta_0=beta_0, dic=dic))
        else:
            plt.title("分配过程  (dp={dp}, k={k}, alpha0={alpha_0}, beta0={beta_0}, DIC={dic})"
                      .format(dp=dp, k=Cluster.k, alpha_0=alpha_0, beta_0=beta_0, dic=dic))
        plt.plot([i + 1 for i in range(niterate)],
                 num_cluster, 'bo-', markersize=2)
        plt.xlabel("迭代次数")
        plt.ylabel("剩余类别数")
        plt.savefig(path)
    # cluster序号按均值排序和标号
    # 重新编排样本对应cluster
    info = [Info() for _ in range(len(clusters))]
    id2id = dict[int, int]()
    idx = 0
    for i in clusters:
        cluster = clusters[i]
        info[idx].mu = cluster.mu
        info[idx].sigma = 1.0 / cluster.inv_sigma
        info[idx].size = cluster.size
        info[idx].idx = idx
        id2id[i] = idx
        idx += 1
    for i in range(ndata):
        for j in range(len(data[i])):
            idx2cluster[i][j] = id2id[idx2cluster[i][j]]
    id2id.clear()
    info.sort(key=lambda x: x.mu)
    for i in range(len(info)):
        id2id[info[i].idx] = i
        info[i].idx = i
    for i in range(ndata):
        for j in range(len(data[i])):
            idx2cluster[i][j] = id2id[idx2cluster[i][j]]
    return info, idx2cluster

