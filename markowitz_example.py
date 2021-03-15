import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import pandas_datareader.data as web
import pandas_datareader.stooq as stooq
import skopt

np.random.seed(123)
pd.options.display.width = 0
gen = skopt.sampler.sobol.Sobol()

def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)

def rand_sobol(n):
    return gen.generate(dimensions = ([(0.,1.)]*n), n_samples=1, random_state=None)

def get_gdrive_data(tp = 'btc'):
    if tp == 'xrp':
        link = r"https://drive.google.com/u/0/uc?id=1cx0wSRPKpsz3pDLQYb7ZG6foBETPAaFZ&export=download"
    if tp == 'btc':
        link = r"https://drive.google.com/u/0/uc?id=1EiyTdW8tKBJHURJ_OWyct3IqvHQblS0A&export=download"
    return pd.read_csv(link)

def get_data(choice):

    if choice == 'Stocks':
        tickers = ['^DJI', 'TSLA', 'AAPL', 'JPM','DAX']
    elif choice == 'ETF':
        tickers = ['LOCK.UK', 'RBTX.UK', 'DGIT.UK', 'ECAR.UK', 'IH2O.UK', 'DRDR.UK']
    elif choice == 'BTC':
        tickers = ['ETH','LTC',] # 'BTC.V', 'XRP.V' 'DOGE'
        #btc = pd.read_csv(r"D:\DLS\Data Dump\btc.v.csv")
        btc = get_gdrive_data('btc')
        btc['Date'] = btc['Date'].map(pd.to_datetime)
        btc.set_index('Date',inplace=True)
        #xrp = pd.read_csv(r"D:\DLS\Data Dump\xrp.v.csv")
        xrp = get_gdrive_data('xrp')
        xrp['Date'] = xrp['Date'].map(pd.to_datetime)
        xrp.set_index('Date',inplace=True)

    data = [ web.DataReader(x, 'stooq') for x in tickers ]

    df = pd.concat( [datai.Close for datai in data] +([btc.Close, xrp.Close] if choice == 'BTC' else []) , axis = 1 )
    df.columns = tickers + (['BTC','XRP'] if choice == 'BTC' else [])
    df = df.dropna().sort_index()
    return df

####
def generate_opt_portfolios(df,nq = 20, nrange = 2000, qMaxScale = 2):
    """
    w^{T}\Sigma w-q*R^{T}w
    Brute-Force Solution
    """
    returns = np.log(df/df.shift(1))
    R=(df.diff()[1:])/df.iloc[0]*100 # In percentage terms, per-day
    R = df.pct_change().dropna()
    w = np.array( [1.0/len(df.columns)]*len(df.columns) )
    ER = R.mean()
    PR = (w * ER).sum()
    Sigma = R.cov()
    VAR = np.dot( np.dot( w , Sigma ) , w )

    def f(w,q):
        var = np.dot(np.dot(w, Sigma), w) *252
        PR = (w*ER).sum() * 252
        sharpe = PR/np.sqrt(var)
        return var, PR, var - q*PR, sharpe

    q_max = np.ceil( np.matrix(Sigma).diagonal().max() )* qMaxScale
    q_range = [x/nq*q_max for x in range(nq)]

    res = []
    for ii in range(nrange):
        t_w = rand_weights(len(df.columns))
        for q in q_range:
            var, pr, fmin, sharpe = f(t_w,q)
            res.append( [fmin, q, var, pr, sharpe, *t_w] )

    df_res = pd.DataFrame(res)
    df_res.columns = ['fmin','q', 'var', 'pr','sharpe'] + list(df.columns)
    df_res.sort_values(by='fmin')

    df_res[['var','pr']].plot.scatter('var','pr')

    idx_min = df_res.groupby(['q'])['fmin'].transform(min) == df_res['fmin']
    idx_sharpe = df_res.groupby(['q'])['sharpe'].transform(max) == df_res['sharpe']
    opt_portfolios = df_res[idx_min]
    opt_sharpe = df_res[idx_sharpe]
    opt_portfolios = opt_portfolios.append(opt_sharpe)
    opt_portfolios = opt_portfolios[['var', 'pr','sharpe'] + list(df.columns)].drop_duplicates()
    opt_portfolios = opt_portfolios.reset_index(drop=True)
    return opt_portfolios

### BackTest
def generate_pfolios(df,opt_portfolios):
    # Notional
    pfolios = df.copy()
    pfolios = pfolios/pfolios.iloc[0]
    notional_weights = 1/df.iloc[0]
    eqw = 1/float(len(notional_weights))
    pfolio_EQW = (df*notional_weights*eqw).sum(axis=1)
    pfolios['EQUAL_W'] = pfolio_EQW

    # Risk-Weighted
    for i in opt_portfolios.index:
        risk_weights = notional_weights*opt_portfolios.iloc[i,3:]
        pfolio_rw = (df*risk_weights).sum(axis=1)
        pfolios['RW'+str(i)] = pfolio_rw
    return pfolios

# Monthly Returns
def get_monthly_returns(pfolios):
    monthly = pfolios.resample("M").last()
    gains = monthly.diff()
    means = gains.mean()
    std = gains.std()
    sharpe = means/std
    previous_peaks = pfolios.cummax()
    drawdown = ((pfolios - previous_peaks)/previous_peaks).min()*100
    round(drawdown.min(),4)
    monthly_stats = pd.DataFrame( [means, std, sharpe, drawdown ]).round(4)
    monthly_stats.index = ['MeanGain','STD','Sharpe','MaxDrawdown']
    return monthly_stats

def run():
    df = get_data(choice='BTC')
    df = df[df.index > pd.to_datetime(datetime.date(2019, 1, 1))]
    opt_portfolios = generate_opt_portfolios(df)
    pfolios = generate_pfolios(df, opt_portfolios)
    (pfolios - pfolios.iloc[0]).plot()
    print(get_monthly_returns(pfolios))

# Stress-Test
def run_stress_test(data, tickers, choice):
    df = get_data(choice)
    btc_crash_one = df[(df.index>pd.to_datetime(datetime.date(2018,1,1)))&(df.index<pd.to_datetime(datetime.date(2019,1,1)))]
    btc_crash_two = df[(df.index>pd.to_datetime(datetime.date(2019,7,1)))&(df.index<pd.to_datetime(datetime.date(2020,3,1)))]

    btc_crash_one_pfolios = generate_pfolios(btc_crash_one,opt_portfolios)
    btc_crash_one_pfolios.plot()
    get_monthly_returns(btc_crash_one_pfolios)

    btc_crash_two_pfolios = generate_pfolios(btc_crash_two, opt_portfolios)
    btc_crash_two_pfolios.plot()
    get_monthly_returns(btc_crash_two_pfolios)
