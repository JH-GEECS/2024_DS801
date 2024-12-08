import gymnasium as gym
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import numpy as np
import torch

import const


# gym.Env를 상속해서 새로운 환경을 만들어야 한다.
class MarketEnv(gym.Env):
    """
    action space는 이용 가능한 portfolio n개에 대해서, 각각의 비율을 조절하는 것이다.
    해당 action을 취하면 결과로 취득할 수 있는 것이 differential sharpe ratio와 
    
    sharpe ratio: risk-adjusted return
    
    첫 거래일, 바로 다음 거래일, 최종 거래일
    
    """
    def __init__(self,
                 lookback_T: int,
                 sharpe_eta: float,
                 asset_definition: Dict[str, str],
                 market_df: pd.DataFrame,
                 initial_cash: float,
                 deterministic: bool = False,
                 debug: bool = False
                 ):
        
        self.debug = debug
        self.deterministic = deterministic
        
        self.initial_cash = initial_cash
        
        # some definiitions of assets
        self.market_df: pd.DataFrame = market_df
        self.idx2asset = {i: asset for i, asset in enumerate(asset_definition.keys())}
        self.idx2asset[len(asset_definition)] = 'cash'
        self.asset2idx = {asset: i for i, asset in enumerate(asset_definition.keys())}
        self.asset2idx['cash'] = len(asset_definition)

        self.lookback_T = lookback_T
        self.business_days = len(market_df)
        self.num_securities = len(asset_definition)
        self.num_all_asset = self.num_securities + 1  # including cash
        self.share_eta = sharpe_eta
        
        # RL agent 학습을 위한 state 정보
        
        # time stamp 만들기
        self.time_step = 0
        # [S_1, S_2, ..., S_n], 나중에 debug하기 쉽도록 전체 정보에 대한 저장
        self.overall_state = np.zeros(
            (self.business_days, self.num_all_asset, self.lookback_T)
        )  # to handle cash
        
        # portfolio 가치 산정을 위한 부분
        # 현금 이외의 금액과 현금 금액
        self.portfolio_ac = np.zeros((self.business_days, 2))
        # 실 거래를 위한 quantized shares
        self.portfolio_shares = np.zeros((self.business_days, self.num_securities))
        # sharpe ratio 계산을 위한 부분
        self.portfolio_return = np.zeros((self.business_days, 1))
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_all_asset, self.lookback_T),
            dtype=np.float32
        )
        
        self.action_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.num_all_asset,),
            dtype=np.float32
        )
        # n개의 자산에 대해서 예측을 하되, cash의 경우에는 1 - \sum_ i w_i로 계산
        
        ### example
        # # Define the agent and target location; randomly chosen in `reset` and updated in `step`
        # self._agent_location = np.array([-1, -1], dtype=np.int32)
        # self._target_location = np.array([-1, -1], dtype=np.int32)
        # # Observations are dictionaries with the agent's and the target's location.
        # # Each location is encoded as an element of {0, ..., `size`-1}^2
        # self.observation_space = gym.spaces.Dict(
        #     {
        #         "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
        #         "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
        #     }
        # )
        # # We have 4 actions, corresponding to "right", "up", "left", "down"
        # self.action_space = gym.spaces.Discrete(4)
        # # Dictionary maps the abstract actions to the directions on the grid
        # self._action_to_direction = {
        #     0: np.array([1, 0]),  # right
        #     1: np.array([0, 1]),  # up
        #     2: np.array([-1, 0]),  # left
        #     3: np.array([0, -1]),  # down
        # }
        ###
    
    def _get_asset_columns(self, suffix: str):
        """
        cash를 제외한 column들을 가져오기 위함
        """
        asset_columns = [f"{asset}_{suffix}" for asset in self.idx2asset.values()]
        del asset_columns[-1]
        
        return asset_columns
        
        
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        여기서 overall obersevation를 잘 정의 해주도록 한다.
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        initial_cash = self.initial_cash
        
        # 환경 0으로 초기화
        self.overall_state = np.zeros(
            (self.business_days, self.num_all_asset, self.lookback_T)
        )
        self.portfolio_ac = np.zeros((self.business_days, 2))
        self.portfolio_shares = np.zeros((self.business_days, self.num_securities))
        self.portfolio_return = np.zeros((self.business_days, 1))
        
        # 처음 거래 시작일, portfolio의 처음은 당연하게도 현금만 운용 가능
        # 여기에서 randomness를 넣어도 될 듯 하다
        # 언제부터 시작할 지, seed에 의해서 random하게 결정
        if not self.deterministic:
            start = np.random.randint(self.lookback_T - 1, self.business_days - 1)
        else:
            start = self.lookback_T - 1
        self.business_start_idx = self.time_step = start
        
        # RL environment 초기화
        asset_columns = self._get_asset_columns("log_r")
        
        for t in range(self.lookback_T-1,self.business_days, 1):
            # 당일 전부터 lookback_T일 전까지의 데이터를 저장
            self.overall_state[t, :self.num_securities, 1:] = self.market_df[asset_columns][(t-(self.lookback_T-1)):t][::-1].to_numpy().T
            
            # 현금과 변동성 정보
            self.overall_state[t, -1][1:3] = self.market_df[["SPX_vol20_normalized", "SPX_vol20_div_vol60_normalized"]][t-1:t].to_numpy()
            self.overall_state[t, -1][3:] = self.market_df[["VIX_close_normalized"]][t-(self.lookback_T-3):t][::-1].to_numpy().T
        self.overall_state[:(self.business_start_idx + 1), -1, 0] = 1.0  # cash

        # portfolio 초기화
        # T-1부터 거래를 시작해야함, 실제로는 T일에 해당하는 거래 data를 활용해야 함
        self.portfolio_ac[:(self.business_start_idx + 1), 1] = initial_cash
        
        ### example
        # Choose the agent's location uniformly at random
        # self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        # # We will sample the target's location randomly until it does not coincide with the agent's location
        # self._target_location = self._agent_location
        # while np.array_equal(self._target_location, self._agent_location):
        #     self._target_location = self.np_random.integers(
        #         0, self.size, size=2, dtype=int
        #     )
        ### example

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _get_obs(self):
        # return {"agent": self._agent_location, "target": self._target_location}
        return self.overall_state[self.time_step]
           
        # agent는 특정 시점일때의 정보에만 접근가능 하도록 하기
        
    
    def _get_info(self):
        """
        여기는 internal 분석용으로 쓰면 될 듯 하다
        여기서는 portfolio의 가치와 shares 반환하면 검산할 때 편할 듯
        """
        return {
            "current_step": self.time_step,
            "port_val_sum": self.portfolio_ac[self.time_step].sum(),
            "port_val": self.portfolio_ac[self.time_step],
            "port_shares": self.portfolio_shares[self.time_step],
        }
    
    def _compute_clip(self, action: np.ndarray):
        # action model이 예측한 raw weights
        
        columns = self._get_asset_columns("close")
        current_asset_prices = self.market_df[columns][self.time_step:self.time_step+1].to_numpy()
        current_securities_sum = np.sum(self.portfolio_shares[self.time_step-1] * current_asset_prices)
        current_portfolio_sum = current_securities_sum + self.portfolio_ac[self.time_step-1, 1]
        
        self.portfolio_shares[self.time_step, :] = np.floor((action * current_portfolio_sum) / current_asset_prices)
        
        self.portfolio_ac[self.time_step, 0] = np.sum(self.portfolio_shares[self.time_step, :] * current_asset_prices)  # 새로운 비중으로 계산된 securities
        self.portfolio_ac[self.time_step, 1] = current_cash = current_portfolio_sum - np.sum(self.portfolio_shares[self.time_step, :] * current_asset_prices)  # 새로운 비중으로 계산된 cash
        
        self.portfolio_return[self.time_step] = (self.portfolio_ac[self.time_step].sum() - self.portfolio_ac[self.time_step-1].sum()) / self.portfolio_ac[self.time_step-1].sum()
        
        # import ipdb; ipdb.set_trace()
        
        self.overall_state[self.time_step, :self.num_securities, 0] = (self.portfolio_shares[self.time_step, :] * current_asset_prices) / current_portfolio_sum
        self.overall_state[self.time_step, -1, 0] = current_cash / current_portfolio_sum
    
    def _compute_reward(self):
        """
        Differntial Sharpe Ratio 계산
        
        """
        # 거래 시작일 부터 portfolio의 return을 계산해야 함
        
        A_t1 = self.portfolio_return[self.business_start_idx:self.time_step].mean()
        delta_A_t = self.portfolio_return[self.time_step] - A_t1
        A_t = A_t1 + self.share_eta * delta_A_t

        B_t1 = (self.portfolio_ac[self.business_start_idx:self.time_step, 0] ** 2).mean()
        delta_B_t = (self.portfolio_return[self.time_step] ** 2) - (B_t1)
        B_t = B_t1 + self.share_eta * delta_B_t

        # import ipdb; ipdb.set_trace()
        if abs(B_t1 - A_t1**2) < const.epsilon: 
            diff_sharpe_ratio = 0.0
        else:
            diff_sharpe_ratio = ((B_t1 * delta_A_t - (1/2)*A_t1*delta_B_t)/(B_t1 - A_t1**2)**(3/2))
        
        return np.squeeze(diff_sharpe_ratio)

    @classmethod
    def softmax(cls, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    
    def _process_action(self, action):
        """
        model의 raw 출력을 softmax를 취해주고, 자산에 대한 비율만 넘겨주기
        """
        return self.softmax(action)[:self.num_securities]
        
    def step(self, action):
        """
        action은 각 portfolio 배분 비율로 하면 될듯 하다.
        
        action을 취했을 때, 어떠한 결과를 얻어야 하는가?
        
        PortVal_t = \sum P_{i,t} \times shares_{i, t-1} + cash_{t-1}
        
        shares 
        """
        self.time_step += 1
        
        # 여기서 model의 raw actio을 processing을 해주어야 겠다.
        self._compute_clip(self._process_action(action))  # 자산 가치 계산
        
        reward = self._compute_reward()
        observation = self._get_obs()
        info = self._get_info()
        
        # An environment is completed if and only if the agent has reached the target
        terminated = True if self.time_step == self.business_days - 1 else False
        truncated = False

        # # Map the action (element of {0,1,2,3}) to the direction we walk in
        # direction = self._action_to_direction[action]
        # # We use `np.clip` to make sure we don't leave the grid bounds
        # self._agent_location = np.clip(
        #     self._agent_location + direction, 0, self.size - 1
        # )
        
        return observation, reward, terminated, truncated, info