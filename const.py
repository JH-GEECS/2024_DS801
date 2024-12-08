from dataclasses import dataclass
import torch

# Dataset const
SPLIT = ["train", "val", "test"]

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

# PPO envs
@dataclass
class PPOEnvConst:
    N_ENVS = 10
    # TOTAL_TIMESTEPS = 7_500_000  # 7.5M steps per round
    N_STEPS = 252 * 3 * N_ENVS   # rollout buffer size
    BATCH_SIZE = 1260            # 252 * 5
    N_EPOCHS = 16
    GAMMA = 0.9
    GAE_LAMBDA = 0.9
    CLIP_RANGE = 0.25
    LEARNING_RATE_START = 3e-4
    LEARNING_RATE_END = 1e-5
    
@dataclass
class PPOArch:
    net_arch = [64, 64]
    act_fn = torch.nn.Tanh
    log_std_init = -1.0
3