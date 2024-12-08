SNP_INDICES_ASSETS = {
    "SPLRCT": "S&P 500 Information Technology",  # done
    "SPLRCL": "S&P 500 Telecom Services",  # done
    "SPLRCM": "S&P 500 Materials",  # done
    "SPLRCREC": "S&P 500 Real Estate",  # 20020101부터 사용가능  # done
    "SPLRCS": "S&P 500 Consumer Staples",  # done
    "SPSY": "S&P 500 Financials",  # done
    "SPNY": "S&P 500 Energy",  # done
    "SPXHC": "S&P 500 Health Care",  # done, # 중간에 정보 손실 interpolation 해버리기
    "SPLRCD": "S&P 500 Consumer Discretionary",  # done
    "SPLRCI": "S&P 500 Industrials",  # done
    "SPLRCU": "S&P 500 Utilities",  # done
}

LOOKBACK_T = 60  # follows the Sood et al. (2023)

SHARPE_ETA = 1 / 252  # daily sharpe ratio

### computation const
epsilon = 1e-8