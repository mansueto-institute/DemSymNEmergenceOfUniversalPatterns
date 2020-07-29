import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gsp
import matplotlib.colors as cols
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.close('all')
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)

# https://www.census.gov/population/www/documentation/twps0027/twps0027.html
sourceTemp = '<pathToSource>'

df = pd.DataFrame(columns=['Year', 'Rank', 'MetroArea', 'Pop'])

for i in range(21):
    cdf = pd.read_csv(sourceTemp.format(i+2))
    cdf.columns = ['Rank', 'MetroArea', 'Pop']
    cdf['Year'] = np.ones(np.shape(cdf.Rank.shape[0])) * (1790 + 10*i)
    cdf.Year = cdf.Year.apply(int)
    cdf.Pop = cdf.Pop.apply(int)
    df = df.append(cdf, ignore_index=True, sort=False)

pdf = df.pivot(index='Rank', columns='Year', values='Pop')
# pdf = pdf.iloc[:20, :]
mdf = pdf.copy().apply(pd.to_numeric)
mdf = mdf/np.nansum(mdf.values, axis=0)
mdf.columns = map(np.int, mdf.columns)
y = 1/np.arange(1, mdf.shape[0]+1)
zipf = y/y.sum()

f0 = plt.figure(figsize = (8,5))
spec0 = gsp.GridSpec(ncols = 1, nrows = 1)
ax0 = f0.add_subplot(spec0[0,:])
colors = plt.cm.gist_earth(np.linspace(0,0.8,mdf.shape[1]))

sumy = np.zeros((mdf.columns.shape[0], y.shape[0]))
dkls = []
div = np.zeros((mdf.columns.shape[0], y.shape[0]))
for i, col in enumerate(mdf.columns):
    cy = np.sort(mdf[col])[::-1]
    y = cy[~np.isnan(cy)]
    y = y*np.sum(zipf[:y.shape[0]])
    div[i:, :y.shape[0]] += 1

    dkls.append(stats.entropy(y, qk=zipf[:y.shape[0]]))
    x = np.arange(y.shape[0])+1
    if i%5 == 0:
        ax0.plot(x, y, '--', alpha=0.5, color=cols.to_hex(colors[i]), label=col)
    else:
        ax0.plot(x, y, '--', alpha=0.5, color=colors[i])
    sumy[i:, :y.shape[0]] += y
    ax0.set_xscale('log')
    ax0.set_yscale('log')

y = zipf
y = y/y.sum()
x = np.arange(1, mdf.shape[0]+1)
ax0.plot(x, y, '--', lw=3, alpha=1, label='Zipf')

sumy = sumy/ div
sumy = np.array([sy/np.nansum(sy) for sy in sumy])
ax0.plot(x, sumy[-1,:], '-r', lw=3, label='Mean')
ax0.set_xlabel('Rank')
ax0.set_ylabel('$N_i/N_T$')
ax0.legend(loc=3, prop={'size':13})
ax0.set_ylim(1.4e-3, ax0.get_ylim()[1])
plt.tight_layout()
f0i = inset_axes(plt.gca(), width = '40%', height='28%', loc=1, borderpad=1.15)
x, y = mdf.columns, np.array(dkls)
f0i.plot(x, y)
kly = [stats.entropy(sy[~np.isnan(sy)], qk=zipf[~np.isnan(sy)]) for sy in sumy]
f0i.plot(x, kly, 'r')
plt.setp(f0i.get_xticklabels(), fontsize=13)
plt.setp(f0i.get_yticklabels(), fontsize=13)
f0i.set_xlabel('Year', fontsize = 15)
f0i.set_ylabel('$D_{KL}(P|P_z)$', fontsize = 15)
f0i.set_xticks(np.round(np.linspace(x.min(), x.max(), 5)))
f0i.set_yticks(np.linspace(0, 0.05, 3))
f0i.set_ylim(0, f0i.get_ylim()[1])

plt.show()



