
from dataclasses import dataclass
from enum import Enum
import logging
from utils import extract_price, extract_response
from llm_setup import act_gpt


class Action(Enum):
    ANNOUNCE = 'announce'
    RESPOND = 'respond'


class AgentType(Enum):
    BUYER = 'buyer'
    SELLER = 'seller'


class AnnouncementType(Enum):
    BUY = 'buy'
    SELL = 'sell'


class Outcome(Enum):
    ACCEPTED = 'accepted'
    REJECTED = 'rejected'


@dataclass(frozen=True)
class AgentLLMConfig:
    model: str
    max_tokens: int
    temperature: float


@dataclass(frozen=True)
class AgentPromptConfig:
    main_prompt: str
    response_prompt: str
    announcement_prompt: str


class Agent:
    def __init__(self, id: int, reservation_price: float, type: AgentType, 
                 prompt_config: AgentPromptConfig, llm_config: AgentLLMConfig,
                 n_rounds: int, n_iter: int):
        self._id = id
        self._reservation_price = reservation_price
        self._type = type
        self.prompt_config = prompt_config
        self.llm_config = llm_config
        self.n_rounds = n_rounds
        self.n_iter = n_iter
        self.own_history_prompt = ""
        self.own_history_data = []
    
    def render_prompt(self, market_history: str, round: int, iteration: int, 
                      action_prompt: str) -> str:
        prompt = self.prompt_config.main_prompt.format(
            N_ROUNDS=self.n_rounds,
            N_ITER=self.n_iter,
            reservation_price=self._reservation_price,
            market_history=market_history,
            own_history=self.own_history_prompt,
            round=round,
            iteration=iteration,
            action_prompt=action_prompt
        )
        return prompt
    
    def generate_text_with_llm(self, prompt: str) -> str:
        logging.info(f"{self._type.value.capitalize()} with id {self._id} calling the LLM with the prompt: \n{prompt}")
        llm_text = act_gpt(prompt, 
                           self.llm_config.model, 
                           self.llm_config.max_tokens, 
                           self.llm_config.temperature
        )
        logging.info(f"LLM response: {llm_text}")
        return llm_text

    def respond(self, price: float, market_history: str, round: int, iteration: int) -> bool:
        action_prompt = self.prompt_config.response_prompt.format(price=price)
        prompt = self.render_prompt(market_history, round, iteration, action_prompt)
        llm_text = self.generate_text_with_llm(prompt)
        response = extract_response(llm_text)
        return response

    def announce(self, market_history: str, round: int, iteration: int) -> float:
        action_prompt = self.prompt_config.announcement_prompt
        prompt = self.render_prompt(market_history, round, iteration, action_prompt)
        llm_text = self.generate_text_with_llm(prompt)
        price = extract_price(llm_text)
        return price

    def update_own_announcement_history(self, price: float, round: int, iteration: int, accepted: bool):
        outcome = Outcome.ACCEPTED if accepted else Outcome.REJECTED
        if self._type == AgentType.BUYER:
            announcement_type = AnnouncementType.BUY
        else:
            announcement_type = AnnouncementType.SELL
        #TODO: move own history prompt template to yaml config file
        self.own_history_prompt += f"In round {round} at iteration {iteration}, your offer to {announcement_type.value} for ${price:.2f} was {outcome.value}.\n"
        self.update_own_history_data(round, iteration, Action.ANNOUNCE, price, accepted)

    def update_own_responding_history(self, price: float, round: int, iteration: int, accepted: bool):
        outcome = Outcome.ACCEPTED if accepted else Outcome.REJECTED
        if self._type == AgentType.BUYER:
            opposite_announcement_type = AnnouncementType.SELL
        else:
            opposite_announcement_type = AnnouncementType.BUY
        self.own_history_prompt += f"In round {round} at iteration {iteration}, you {outcome.value} an offer to {opposite_announcement_type.value} for ${price:.2f}.\n"
        self.update_own_history_data(round, iteration, Action.RESPOND, price, accepted)

    def update_own_history_data(self, round: int, iteration: int, action: Action, price: float, accepted: bool):
        outcome = Outcome.ACCEPTED if accepted else Outcome.REJECTED
        self.own_history_data.append({
            'round': round,
            'iteration': iteration,
            'action': action.value,
            'price': price,
            'outcome': outcome.value
        })
