# Written by Xuyang Yan.
# xyan@aggies.ncat.edu.
#
# ------------------------------------------------------------------------------
import numpy as np
import math


def fitness_cal(DisC, Nc, label, StdF, gamma):
    fitness = np.zeros(np.shape(DisC)[0])
    for i in range(Nc):
        TempSum = 0
        for j in range(Nc):
            if j != i:
                D = DisC[i, j]
                TempSum = TempSum + (math.exp(- (D ** 2) / StdF)) ** gamma
        fitness[i] = TempSum
    return fitness


def Pseduo_Peaks(DisC, label, fitness, StdF, gamma):
    # Search Stage of Pseduo Clusters at the temporal sample space
    NeiRad = 0.1 * np.max(DisC)  # 0.83
    i = 0
    marked = []
    N = np.shape(DisC)[0]
    C_Indices = np.arange(0, N)  # The pseduo Cluster label of features
    PeakIndices = []
    Pfitness = []
    co = []
    F = fitness
    while True:
        sort_id = np.argsort(F)[::-1]

        if np.max(F) < 1:
            for j in range(np.shape(sort_id)[0]):
                if sort_id[j] in PeakIndices:
                    continue
                else:
                    PeakIndices.append(sort_id[j])
                    Pfitness.append(F[sort_id[j]])
                    break

        else:
            PeakIndices.append(np.argmax(F))
            Pfitness.append(np.max(F))

        indices = NeighborSearch(DisC, label, PeakIndices[i], marked, NeiRad)

        # C_Indices[indices] = PeakIndices[i]
        if len(indices) == 0:
            indices = [PeakIndices[i]]

        co.append(len(indices))  # Number of samples belong to the current
        # identified pseduo cluster
        marked = np.concatenate(([marked, indices]))

        # Fitness Proportionate Sharing
        tempF = Sharing(F, indices)

        F = tempF

        # Check whether all of samples has been assigned a pseduo cluster label
        if np.sum(co) >= (len(F)):
            break

        i = i + 1  # Expand the size of the pseudo cluster set by 1
    return PeakIndices, Pfitness


def NeighborSearch(DisC, label, P_indice, marked, radius):
    Cluster = []
    N = np.shape(DisC)[0]
    for i in range(N):
        if i not in marked:
            Dist = DisC[i, P_indice]
            if Dist <= radius:
                Cluster.append(i)
    return Cluster


def Sharing(fitness, indices):
    newfitness = fitness
    sum1 = 0
    for j in range(len(indices)):
        sum1 = sum1 + fitness[indices[j]]
    for th in range(len(indices)):
        newfitness[indices[th]] = fitness[indices[th]] / (1 + sum1)
    return newfitness


def Pseduo_Evolve(DisC, PeakIndices, PseDuoF, C_Indices, data, fitness, StdF, gamma, K):
    if len(PeakIndices) <= K:
        return PeakIndices, PseDuoF, C_Indices

    # Initialize the indices of Historical Pseduo Clusters and their fitness values
    HistCluster = PeakIndices
    HistClusterF = PseDuoF
    while True:
        # Call the merge function in each iteration
        [Cluster, Cfitness, F_Indices] = Pseduo_Merge(DisC, HistCluster, HistClusterF, C_Indices, data, fitness, StdF,
                                                      gamma)
        # Check for the stablization of clutser evolution and exit the loop
        if len(Cluster) <= K:
            break
        HistCluster = Cluster
        HistClusterF = Cfitness
        C_Indices = F_Indices
    # Compute final evolved feature cluster information
    FCluster = Cluster
    Ffitness = Cfitness
    C_Indices = F_Indices
    return FCluster, Ffitness, C_Indices


# ----------------------------------------------------------------------------------------------------------
def Pseduo_Merge(DisC, PeakIndices, PseDuoF, C_Indices, data, fitness, StdF, gamma):
    # Initialize the pseduo feature clusters lables for all features
    F_Indices = C_Indices
    ML = []  # Initialize the merge list as empty
    marked = []  # List of checked Pseduo Clusters Indices
    Unmarked = []  # List of unmerged Pseduo Clusters Indices
    for i in range(len(PeakIndices)):
        MinDist = math.inf  # Set the default Minimum distance between two feature clusters as infinite
        MinIndice = -1  # Set the default Neighboring feature cluster indices as negative
        # Check the current Pseduo Feature Cluster has been evaluated or not

        if PeakIndices[i] not in marked:
            for j in range(len(PeakIndices)):
                if j != i:
                    # Divergence Calculation between two pseduo feature clusters
                    D = DisC[PeakIndices[i], PeakIndices[j]]
                    if MinDist > D:
                        MinDist = D
                        MinIndice = j

            if MinIndice >= 0:  # check whether there is a closest indices or not
                if PeakIndices[MinIndice] not in marked:
                    ML.append([PeakIndices[i], PeakIndices[MinIndice]])
                    marked.append(PeakIndices[i])
                    marked.append(PeakIndices[MinIndice])
                else:
                    Unmarked.append(PeakIndices[i])

    NewPI = []
    # Update the pseduo feature clusters list with the obtained mergelist
    for m in range(np.shape(ML)[0]):
        if fitness[ML[m][0]] > fitness[ML[m][1]]:
            NewPI.append(ML[m][0])
        else:
            NewPI.append(ML[m][1])
    # Update the pseduo feature clusters list with pseduo clusters that have not appeared in the merge list
    for n in range(len(PeakIndices)):
        if PeakIndices[n] in Unmarked:
            NewPI.append(PeakIndices[n])

    # Updated pseduo feature clusters information after merging
    FCluster = np.unique(NewPI)
    Ffitness = fitness[FCluster]
    F_Indices = cluster_assign(DisC, FCluster, F_Indices)
    return FCluster, Ffitness, F_Indices


def cluster_assign(DisC, P_indices, F_indices):
    # for i in range(len(F_indices)):
    #     dist = []
    #     for j in range(len(P_indices)):
    #         dist.append(DisC[F_indices[i], P_indices[j]])
    #     F_indices[i] = P_indices[np.argmin(dist)]
    # return F_indices
    for i in range(len(F_indices)):
        if i in P_indices:
            F_indices[i] = i
            continue
        else:
            dist = []
            for j in range(len(P_indices)):
                dist.append(DisC[F_indices[i], P_indices[j]])
            F_indices[i] = P_indices[np.argmin(dist)]
    return F_indices


# --------------------------------------------------------------------------------------------------------------
def density_based(K, label, DisC, label_ref=None):
    StdF = max(np.max(DisC), 1)
    if not np.any(DisC):
        StdF = 1

    if label_ref == None:
        label_ref = np.arange(0, np.shape(DisC)[0])
    else:
        label_ref = np.asarray(label_ref)
    Nc = np.shape(DisC)[0]
    gamma = 5
    fitness = fitness_cal(DisC, Nc, label, StdF, gamma)
    oldfitness = np.copy(fitness)

    PeakIndices, Pfitness = Pseduo_Peaks(DisC, label, fitness, StdF, gamma)
    fitness = oldfitness

    # -----------------------------Modified Section---------------------------#
    # Pseduo Clusters Infomormation Extraction
    PseDuo = PeakIndices  # Pseduo Feature Cluster centers
    PseDuoF = Pfitness  # Pseduo Feature Clusters fitness values
    PseDuoFIndice = np.arange(0, Nc)  # Cluster indices before merge
    PseDuoFIndice = cluster_assign(DisC, PeakIndices, PseDuoFIndice)
    C_Indices = np.copy(PseDuoFIndice)

    # -------------Check for possible merges among pseduo clusters-----------#

    [FCluster, _, C_Indices] = Pseduo_Evolve(DisC, PseDuo, PseDuoF, C_Indices, label, fitness, StdF, gamma, K)

    SF = FCluster

    cluster_info1 = {}  # Cluster information before merging
    cluster_info2 = {}  # Cluster information after merging

    # Extract the cluster information before merging
    for Pi in range(len(PseDuo)):
        temp1 = np.where(PseDuoFIndice == PseDuo[Pi])[0]
        temp1 = temp1.astype(int)
        cluster_info1[Pi] = label_ref[temp1].tolist()

    if len(SF) < K:
        return cluster_info1, cluster_info2

    # Extract the cluster information after merging
    for i in range(len(SF)):
        temp2 = np.where(C_Indices == SF[i])[0]
        temp2 = temp2.astype(int)
        cluster_info2[i] = label_ref[temp2].tolist()

    return list(cluster_info1.values()), list(cluster_info2.values())
