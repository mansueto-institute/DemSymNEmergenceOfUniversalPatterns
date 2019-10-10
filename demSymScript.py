import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as lg
import matplotlib.colors as colors
import matplotlib.cm as cmx
import sys
from scipy import stats
from tqdm import tqdm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.close('all')
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)

if len(sys.argv) != 9:
    print('Usage: python3 #Cities #Years SizeDependentFlows(True/False) SigmaFluctuations AvgOutProb VitalRate InitZipfPop(True/False) NumericalPrecision')
else:
    def enforceBoundaries(pop, oldpop, lowBound, upBound):
        '''
        Enforces the boundary conditions
        '''
        pop[pop > upBound] = oldpop[pop > upBound]
        pop[pop < lowBound] = oldpop[pop < lowBound]

        # correct for right amount of people
        poptmp = pop - lowBound
        opoptmp = oldpop - lowBound
        poptmp = poptmp / poptmp.sum() * opoptmp.sum()

        pop = poptmp + lowBound
        return pop

    def fstEigVec(A):
        '''
        Calculate the leading Eigenvector of the environment
        '''
        eigenValues, eigenVectors = lg.eig(A)
        idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:, idx]
        return np.abs(eigenVectors[:, 0].real)


    nCities = int(sys.argv[1])
    nYears = int(float(sys.argv[2]))
    nonLinFlows = True if sys.argv[3] == 'True' else False
    sigmaNoise = float(sys.argv[4])
    avgOutFrac = float(sys.argv[5])
    vitalRate = float(sys.argv[6])
    zipfInit = True if sys.argv[7] == 'True' else False
    prec = float(sys.argv[8])

    # init
    np.random.seed(23)
    Ainit = np.random.random((nCities, nCities))*0.5 + 1
    Atemp = Ainit - np.diag(np.diag(Ainit))
    Atemp = Atemp / np.mean(Atemp.sum(axis=0)) * avgOutFrac
    Ainit = np.diag(np.ones(nCities) + vitalRate - Atemp.sum(axis=0)) + Atemp

    if zipfInit:
        pop = np.ones(nCities)/range(1, nCities+1)
    else:
        pop = np.random.random(nCities)/2 + 0.5
    pop = pop/np.sum(pop)

    zipfPop = np.ones(nCities)/range(1, nCities+1)
    zipfPop = zipfPop / zipfPop.sum() * pop.sum()

    lowBound = np.min(zipfPop)
    upBound = np.inf
    popInit = pop.copy()
    popInit[popInit < lowBound] = lowBound
    popFrac = popInit / popInit.sum()

    popTraj = [popInit]
    fstEigs = [fstEigVec(Ainit)]
    equiIt = 0

    # time integration
    for y in tqdm(range(nYears)):
        A = Ainit/Ainit.sum(axis=0)
        popFrac = pop/pop.sum()

        # stationary state?
        if y > 5:
            diff = np.sum(np.abs(popTraj[-1] - popTraj[-5]))
            if diff < prec*np.sum(popInit) and equiIt == 0:
                equiIt = y

        if nonLinFlows:
            A = A * popFrac[:, np.newaxis]

        if (sigmaNoise > 0):
            A = A * np.random.lognormal(0, sigmaNoise, A.shape)

        A = A - np.diag(np.diag(A))
        A = A / np.mean(A.sum(axis=0)) * avgOutFrac
        asum = A.sum(axis=0)
        A[:, asum>1] = A[:, asum>1] / asum[asum>1]
        A = np.diag(np.ones(nCities) - A.sum(axis=0)) + A
        oldPop = pop
        popFrac = A @ popFrac
        pop = oldPop.sum() * popFrac
        pop = enforceBoundaries(pop, oldPop, lowBound, upBound)
        popTraj.append(pop/pop.sum())
        fstEigs.append(fstEigVec(A) if sigmaNoise > 0 else fstEigs[0])

    dklZipf= []
    for y in range(nYears):
        cpop = np.sort(popTraj[y]/np.sum(popTraj[y]))[::-1]
        dklZipf.append(stats.entropy(cpop, qk=zipfPop))
    dklEig = []
    step = int(nYears/min(10000, nYears))
    years = range(1, nYears+1, step)
    for y in years:
        dklEig.append(stats.entropy(popTraj[y], qk=fstEigs[y]))

    # plot trajectories with D_KL inset
    f1 = plt.figure(figsize = (8, 5))
    plt.plot(range(1, nYears+2, step), popTraj[::step], alpha=0.4)
    if equiIt > 0:
        plt.gca().axvline(linewidth=2, color='red', x=equiIt, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log' if zipfInit else 'linear')
    plt.xlabel('Years')
    plt.ylabel(r'N_i/N_t')
    f1i = inset_axes(plt.gca(), width = '40%', height='28%', loc=1,
            borderpad=1.15)
    f1i.plot(years, dklEig)
    f1i.set_ylabel(r'$D_{KL}(P | P_{\mathbf{e}_0})$', fontsize = 15)
    f1i.set_xscale('log')
    plt.setp(f1i.get_xticklabels(), fontsize=13)
    plt.setp(f1i.get_yticklabels(), fontsize=13)
    plt.tight_layout()

    # plot trajectories with dists inset
    if equiIt > 0:
        f2 = plt.figure(figsize = (8, 5))
        plt.plot(range(1, nYears+2, step), popTraj[::step], alpha=0.4)
        plt.gca().axvline(linewidth=2, color='red', x=equiIt, alpha=0.3)
        plt.xscale('log')
        plt.yscale('log' if zipfInit else 'linear')
        plt.xlabel('Years')
        plt.ylabel(r'N_i/N_t')
        ax2 = plt.gca()
        f2i = inset_axes(plt.gca(), width = '40%', height='28%', loc=2,
                bbox_to_anchor=(0.06,0,1,1), bbox_transform=ax2.transAxes)
        ranks = np.arange(1, nCities+1)
        f2i.plot(ranks, zipfPop, '--', label='Zipf', lw=2, alpha=0.7)
        f2i.plot(ranks, np.sort(popTraj[equiIt])[::-1], 'r', label='Zipf', lw=2, alpha=0.7)
        f2i.set_xscale('log')
        f2i.set_yscale('log')
        plt.setp(f2i.get_xticklabels(), fontsize=13)
        plt.setp(f2i.get_yticklabels(), fontsize=13)
        plt.tight_layout()

    # plot rank-size
    times = np.logspace(0, np.log10(nYears), 5)
    cm = plt.get_cmap('gist_earth')
    cNorm = colors.LogNorm(vmin=times[0], vmax=times[-1]*100)
    scm = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    ranks = np.arange(1, nCities+1)
    dklZipf= []
    for y in range(nYears):
        cpop = np.sort(popTraj[y]/np.sum(popTraj[y]))[::-1]
        dklZipf.append(stats.entropy(cpop, qk=zipfPop))

    f3 = plt.figure(figsize = (8,5))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Rank')
    plt.ylabel(r'$N_i/N_T$')
    plt.plot(ranks, zipfPop, '--', lw=3, label='Zipf')
    for i, x in enumerate(times):
        y = np.sort(popTraj[int(x)])[::-1]
        label = 'Year {:.0f}'.format(x)
        plt.plot(ranks, y/y.sum(), '--', 0.8, color=scm.to_rgba(x),
                label=label)
    bestPop = np.sort(popTraj[np.argmin(dklZipf)])[::-1]
    bestPop = bestPop/np.sum(bestPop)
    plt.plot(ranks, bestPop, '-r', lw=4, alpha=1, label='Best (Year {:.0f})'.format(np.argmin(dklZipf)))
    plt.legend(prop={'size':13})
    plt.tight_layout()

    # plot D_KL trajectory
    years = np.arange(1, nYears+1, step)
    times = np.logspace(0, np.log10(nYears), 5)
    cm = plt.get_cmap('gist_earth')
    cNorm = colors.LogNorm(vmin=times[0], vmax=times[-1]*100)
    scm = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    f4 = plt.figure(figsize = (8, 5))
    plt.plot(years, dklZipf[::step])
    plt.xscale('log')
    for t in times:
        plt.gca().axvline(linewidth=2, color=scm.to_rgba(t), ls='--', x=t,
                alpha=0.8)
    plt.gca().axvline(linewidth=2, color='red', ls='-',
            x=np.argmin(dklZipf)+1, alpha=0.8)
    plt.xlabel('Years')
    plt.ylabel(r'$D_{KL}(P | P_z)$')
    plt.tight_layout()

    plt.show()

