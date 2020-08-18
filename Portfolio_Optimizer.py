# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 13:12:29 2020

@author: massa

"""

import pandas as pd
import numpy as np
import Stats_pack as sp
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.style.use('ggplot') 

def portfolio_vol_rets(data, n_points = 1000, dc = 252, weights = True):
    """
    Inputs:
    data: dataframe com séries temporais dos preços de determinados ativos
    rfree: série temporal da taxa livre de risco. Para um portfólio BR, usar
    série do CDI que pode ser importada com o módulo I_Database.
    n_points: quantidade de portfolios a serem simulados
    dc: períodos de trading usados na análise.
    ----------------------------------------------------
    Calcula os retornos, volatilidades e pesos de uam determinada quantidade
    de portfólios aleatórios.
    """
    np.random.seed(0)
    ret_arr = np.zeros(n_points)
    vol_arr = np.zeros(n_points)
    aw = np.zeros((n_points, len(data.columns)))
    rets = sp.Log_Ret(data)
    cov = sp.Cov_Matrix(data, periods = dc,ret_type='log')
    for i in tqdm(range(n_points)):
        rw = np.array(np.random.random(len(data.columns)))
        rw = rw/np.sum(rw)
        aw[i] = rw
        ret_arr[i] = np.sum((rets.mean()*rw)* dc)
        vol_arr[i] = np.sqrt(rw.T @ cov @ rw)
    if weights == True:
        return ret_arr,vol_arr, aw
    else:
        return ret_arr,vol_arr

def portfolio_sharpe(rets, vols, rfree):
    """
    Inputs:
    rets: vetor que contém os retornos de portfólios gerados aleatoriamente.
    vols: vetor que contém as volatilidades de portfólios gerados aleatoriamente.
    rfree: série temporal da taxa livre de risco. Para um portfólio BR, usar
    série do CDI que pode ser importada com o módulo I_Database.
    n_points: quantidade de portfolios a serem simulados
    ----------------------------------------------------
    Calcula os sharpes dos portfólios aleatórios e retorna um vetor com estes 
    valores.
    """
    sharpe_arr = np.zeros(len(rets))
    rf = rfree.iloc[-1][rfree.columns]
    for i in tqdm(range(len(rets))):
        sharpe_arr[i] = (rets[i]-rf)/vols[i]        
    return sharpe_arr
    
def plot_frontier(data, rfree,n_points = 1000 ,plot_msp = True, plot_gmv = True):
    """
    Inputs:
    data: dataframe com séries temporais dos preços de determinados ativos
    rfree: série temporal da taxa livre de risco. Para um portfólio BR, usar
    série do CDI que pode ser importada com o módulo I_Database.
    n_points: quantidade de portfolios a serem simulados 
    ----------------------------------------------------
    Calcula os pesos dos ativos que maximizam o Sharpe de um portfólio.
    """
    ret_arr, vol_arr = portfolio_vol_rets(data, n_points = n_points, 
                                          weights=False)
    sharpe_arr = portfolio_sharpe(ret_arr,vol_arr,rfree)    
    plt.scatter(vol_arr, ret_arr, c = sharpe_arr, cmap = 'inferno')
    plt.colorbar(label = 'Indice Sharpe')
    plt.xlabel('Volatilidade Esperada')
    plt.ylabel('Retorno Esperado')
    if plot_msp:
        sharpe_ret = ret_arr[sharpe_arr.argmax()]
        sharpe_vol = vol_arr[sharpe_arr.argmax()]
        plt.scatter(sharpe_vol, sharpe_ret, marker = 'o', s = 50, c = 'r')
    if plot_gmv:
        gmv_ret = ret_arr[vol_arr.argmin()]
        gmv_vol = vol_arr[vol_arr.argmin()]
        plt.scatter(gmv_vol, gmv_ret, marker = 'o', s = 50, c = 'g')
    plt.show()

def max_sharpe_weights(weights, sharpes):
    """
    Inputs:
    weights: vetor que contém pesos aleatórios dos componentes da carteira sele
    cionada.
    sharpes: vetor que contém os índices sharpes das carteiras aleatórias
    ----------------------------------------------------
    Retorna os pesos que maximizam o sharpe da carteira.
    """
    wms = weights[sharpes.argmax()]
    return wms

def gmv_weights(weights, vols):
    """
    Inputs:
    weights: vetor que contém pesos aleatórios dos componentes da carteira sele
    cionada.
    vols: vetor que contém as volatilidades esperadas das carteiras aleatórias
    ----------------------------------------------------
    Retorna os pesos que minimizam a volatilidade de um portfólio.
    """
    wgmv = weights[vols.argmin()]
    return wgmv

def backtest_pfolio(data, rfree, weight_scheme = 'MS', window = None, **kwargs):
    """
    Inputs:
    data: dataframe que contém preços dos ativos para um determinado período.
    rfree: série histórica da taxa livre de risco. Para portfólio brasileiro,
    usar a função 'importa_cdi', do módulo I_Database.
    weight_scheme: flag que indica o método de alocação dos ativos do portfólio,
    i.e, Maximização de Índice Sharpe e Portfólio de Mínima Variância Global.    
    window: indica a frequência de rebalanceamento da estratégia. A janela de re
    balanceamento deve expressa em dias úteis.
    ----------------------------------------------------
    Calcula os retornos de uma determinada estratégia de alocação de ativos.
    """
    if window is not None:
        ret_arr, vol_arr, wgts_arr = [], [], []     
        N = data.shape[0]
        windows = [[round(start), round(start + window)] for 
                    start in np.linspace(0,N-window,round(N/window))]
        rets = [sp.Simp_Ret(data[w[0]:w[1]]) for w in windows]
        rfw = [rfree[w[0]:w[1]] for w in windows]
        for i in range(round(N/window)):
            ret_arr.append([portfolio_vol_rets(data[w[0]:w[1]], **kwargs) 
            for w in windows][i][0])
            vol_arr.append([portfolio_vol_rets(data[w[0]:w[1]], **kwargs) 
            for w in windows][i][1])
            wgts_arr.append([portfolio_vol_rets(data[w[0]:w[1]], **kwargs) 
            for w in windows][i][2])
        ret_pfolio = pd.DataFrame()
        if weight_scheme.upper() == 'MS':
            sharpe_arr,msw_arr = [],[]
            for i in range(round(N/window)):
                sharpe_arr.append(portfolio_sharpe(ret_arr[i],vol_arr[i],rfw[i]))
                msw_arr.append(max_sharpe_weights(wgts_arr[i],sharpe_arr[i]))                
                ret_pfolio = ret_pfolio.append(rets[i]*msw_arr[i])
            return ret_pfolio
        elif weight_scheme.upper() == 'GMV':
            wgmv_arr = []
            for i in range(round(N/window)):
                wgmv_arr.append(gmv_weights(wgts_arr[i],vol_arr[i]))
                ret_pfolio = ret_pfolio.append(rets[i]*wgmv_arr[i])
            return ret_pfolio               
    elif window is None:
        if weight_scheme.upper() == 'MS':
            (rets, vols, weights) = portfolio_vol_rets(data, **kwargs)
            sharpe = portfolio_sharpe(rets, vols, rfree)
            msw = max_sharpe_weights(weights,sharpe)
            ret_pfolio = np.sum((msw*sp.Simp_Ret(data)), axis = 1)
            return ret_pfolio
        elif weight_scheme.upper() == 'GMV':
            (rets, vols, weights) = portfolio_vol_rets(data, **kwargs)            
            wgmv = gmv_weights(weights, vols)
            ret_pfolio = np.sum((wgmv*sp.Simp_Ret(data)), axis = 1)
            return ret_pfolio
        
            