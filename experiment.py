from abc import ABC, abstractmethod
from typing import Tuple
from pathlib import Path
from enum import Enum
import pandas as pd
import numpy as np
import re
import os
import logging

from llm_setup import act_gpt4_test

# NOTE: the bid/offer terminology is very specific to finance, but maybe that is good

EXPERIMENT_ID = 8

# experiment parameters
N_ROUNDS = 5
N_ITER = 10

# configure the logger
logging.basicConfig(
    filename=Path(__file__).parent.resolve() / f"logs/experiment_{EXPERIMENT_ID}.log",
    filemode='w',                   
    level=logging.INFO
)


def extract_price(number: str) -> float:
    """
    Parses the price response from the LLM. Expects a number in the str format.
    """
    clean_number = number.strip()
    try:
        clean_number_float = float(clean_number)
        return clean_number_float
    except ValueError:
        logging.info(f"Could not parse '{clean_number}' as float.")
        return None 
    

def extract_response(text: str) -> bool:
    """
    Extracts a boolean indicator of whether a deal is accepted from a textual response.
    """
    if "yes" in text.lower():
        return True
    else:
        return False

class Action(Enum):
    ANNOUNCE = 'announce'
    RESPOND = 'respond'


# cleaner design: implement respond/announce method in the parent class
# and let the child classes only differ in the prompt generation

class Agent(ABC):
    def __init__(self, id: int, reservation_price: float):
        self._reservation_price = reservation_price
        self._id = id
        self.own_history_prompt = ""
        self.own_history_data = []

    @abstractmethod
    def respond(self, price: float, history: str, round: int, iteration: int) -> bool: ...

    @abstractmethod
    def announce(self, history: str, round: int, iteration: int) -> float: ...

    def update_own_history_data(self, round: int, iteration: int, action: Action, price: float, accepted: bool):
        outcome = "accepted" if accepted else "rejected"
        self.own_history_data.append({
            'round': round, 
            'iteration': iteration, 
            'action': action.value,
            'price': price,
            'outcome': outcome
        })


class Buyer(Agent):

    instructions = f"""
    You are a buyer participating in a market for a good you need. Your task is to buy a unit of the good at the lowest possible price but no higher than your reservation price.
    Your reservation price is known to you and only you.
    Under no condition can you buy above your reservation price. 
    However, buying at a price equal to your reservation price is acceptable and preferred than not buying at all.
    
    There are 11 sellers and 11 buyers (including you) in the room.
    Any buyer or seller is free at any time to raise his hand and make a verbal offer to buy/sell.
    Any buyer or seller is free to accept anb in th offer, in whicand h case a binding contract has been formed, the transaction occurs and the buyer and seller drop out of the market (no longer permitted to do anything for the remainder of that ronud).

    There will be {N_ROUNDS} rounds.

    Each round, you want to buy an additional unit of the good and are able to transact irrespective of whether you transacted in the previous round.
    Each round, a maximum of {N_ITER} transactions can be made. You can only make one transaction per round.
    \n
    """    

    def __init__(self, id: int, reservation_price: float):
        super().__init__(id, reservation_price)

    def respond(self, price: float, history: str, round: int, iteration: int) -> bool:
        prompt = self.instructions + f"Your reservation price is {self._reservation_price}.\n" + "Market history:\n" + history + "History of your actions:\n" + self.own_history_prompt + f"This is round {round}/{N_ROUNDS} iteration {iteration}/{N_ITER}. Someone is offering to sell at ${price:.2f}. Do you buy? Only answer with a yes or no."
        # call the LLM with the prompt and get the response
        logging.info(f"Buyer with id {self._id} calling the LLM with the prompt: \n{prompt}")
        response_text = act_gpt4_test(prompt)
        logging.info(f"LLM response: {response_text}")
        response = extract_response(response_text)
        return response
        
    def announce(self, history: str, round: int, iteration: int) -> float:
        prompt = self.instructions + f"Your reservation price is {self._reservation_price}.\n" + "Market history:\n" + history + "History of your actions:\n" + self.own_history_prompt + f"This is round {round}/{N_ROUNDS} iteration {iteration}/{N_ITER}. Do you want to announce a bid to buy? If so, what is your bid price? Answer only with a number."
        # call the LLM with the prompt and get the response
        logging.info(f"Buyer with id {self._id} calling the LLM with the prompt: \n{prompt}")
        announcement_text = act_gpt4_test(prompt)
        logging.info(f"LLM reponse: {announcement_text}")
        price = extract_price(announcement_text)
        return price
    
    def update_own_announcement_history(self, price: float, round: int, iteration: int, accepted: bool):
        outcome = "accepted" if accepted else "rejected"
        self.own_history_prompt += f"In round {round} at iteration {iteration}, your offer to buy for ${price:.2f} was {outcome}.\n"
        self.update_own_history_data(round, iteration, Action.ANNOUNCE, price, accepted)
    
    def update_own_responding_history(self, price: float, round: int, iteration: int, accepted: bool):
        outcome = "accepted" if accepted else "rejected"
        self.own_history_prompt += f"In round {round} at iteration {iteration}, you {outcome} an offer to sell for ${price:.2f}.\n"
        self.update_own_history_data(round, iteration, Action.RESPOND, price, accepted)
            

class Seller(Agent):

    instructions = f"""
    You are a seller participating in a market for a good you need to sell. Your task is to sell a unit of the good at the highest possible price but no lower than your reservation price.
    Your reservation price is known to you and only you.
    Under no condition can you sell below your reservation price. 
    However, selling at a price equal to your reservation price is acceptable and preferred than not selling at all.
    
    There are 11 buyers and 11 sellers (including you) in the room.
    Any buyer or seller is free at any time to raise his hand and make a verbal offer to buy or sell.
    Any buyer or seller is free to accept an offer, in which case a binding contract has been formed, the transaction occurs and the buyer and seller drop out of the market (no longer permitted to do anything for the remainder of that round).

    There will be {N_ROUNDS} rounds.

    Each round, you receive an additional unit of the good and are able to transact irrespective of whether you transacted in the previous round.
    Each round, a maximum of {N_ITER} transactions can be made. You can only make one transaction per round.
    \n
    """

    def __init__(self, id: int, reservation_price: float):
        super().__init__(id, reservation_price)

    def respond(self, price: float, history: str, round: int, iteration: int) -> bool:
        prompt = self.instructions + f"Your reservation price is {self._reservation_price}.\n" + "Market history:\n" + history + "History of your actions:\n" + self.own_history_prompt + f"This is round {round}/{N_ROUNDS} iteration {iteration}/{N_ITER}. Someone is offering to buy at ${price:.2f}. Do you sell? Only answer with a yes or no."
        # call the LLM with the prompt and get the response
        logging.info(f"Seller with id {self._id} calling the LLM with the prompt: \n{prompt}")
        response_text = act_gpt4_test(prompt)
        logging.info(f"LLM response: {response_text}")
        response = extract_response(response_text)
        return response
    
    def announce(self, history: str, round: int, iteration: int) -> float:
        prompt = self.instructions + f"Your reservation price is {self._reservation_price}.\n" + "Market history:\n" + history + "History of your actions:\n" + self.own_history_prompt + f"This is round {round}/{N_ROUNDS} iteration {iteration}/{N_ITER}. Do you want to announce an offer to sell? If so, what is your asking price? Answer only with a number."
        # call the LLM with the prompt and extract the response price
        logging.info(f"Seller with id {self._id} calling the LLM with the prompt: \n{prompt}")
        announcement_text = act_gpt4_test(prompt)
        logging.info(f"LLM reponse: {announcement_text}")
        price = extract_price(announcement_text)
        return price
    
    def update_own_announcement_history(self, price: float, round: int, iteration: int, accepted: bool):
        outcome = "accepted" if accepted else "rejected"
        self.own_history_prompt += f"In round {round} at iteration {iteration}, your offer to sell for ${price:.2f} was {outcome}.\n"
        self.update_own_history_data(round, iteration, Action.ANNOUNCE, price, accepted)
    
    def update_own_responding_history(self, price: float, round: int, iteration: int, accepted: bool):
        outcome = "accepted" if accepted else "rejected"
        self.own_history_prompt += f"In round {round} at iteration {iteration}, you {outcome} an offer to buy for ${price:.2f}.\n"
        self.update_own_history_data(round, iteration, Action.RESPOND, price, accepted)


def main():

    # generate reservation prices for the buyers and sellers
    # symmetric for now
    buyers_reservation_prices = np.round(np.linspace(0.8, 3.2, 11), 2)
    sellers_reservation_prices = np.round(np.linspace(0.8, 3.2, 11), 2)
    logging.info(f"Buyers reservation prices: {buyers_reservation_prices}")
    logging.info(f"Sellers reservation prices: {sellers_reservation_prices}")

    # initialise the agents (buyers and sellers with symmetric res prices)
    agents = []
    for id, res_price in enumerate(buyers_reservation_prices):
        agents.append(Buyer(id, res_price))
    for id, res_price in enumerate(buyers_reservation_prices):
        agents.append(Seller(id, res_price))

    # conduct each round of the expeirment seqeuentially
    print("Running the experiment...")
    history = ""
    data_iterations = []
    for round in range(1, N_ROUNDS+1):
        remaining_agents = agents.copy()

        # solicit announcements from all agents at each iteration 
        for iteration in range(1, N_ITER+1):
            # reset the iteration info
            transaction_made = False
            announcement_made = False
            announcement_type = ""
            responding_agent_id = None
            announcing_agent_id = None
            announcing_agent_reservation_price = None
            responding_agent_reservation_price = None
            
            # shuffle the agents order
            np.random.shuffle(remaining_agents)

            # solicit a price announcement
            logging.info("Prompting agents for an announcement...")
            for i, announcing_agent in enumerate(remaining_agents):
                price = announcing_agent.announce(history, round, iteration)
                if price is not None:
                    announcement_made = True
                    if isinstance(announcing_agent, Seller):
                        announcement_type = "sell"
                    else:
                        announcement_type = "buy"
                    logging.info(f"An announcement to {announcement_type} for ${price} was made by agent {announcing_agent._id} at iteration {iteration}.")
                    announcing_agent_id = announcing_agent._id
                    announcing_agent_reservation_price = announcing_agent._reservation_price

                    # obtain a response to the announcement
                    logging.info("Prompting agents for a response to the announcement...")
                    for j, responding_agent in enumerate(remaining_agents):
                        if (isinstance(responding_agent, Seller) and announcement_type == "buy") or (isinstance(responding_agent, Buyer) and announcement_type == "sell"):
                            response = responding_agent.respond(price, history, round, iteration)
                            responding_agent.update_own_responding_history(price, round, iteration, accepted=response)
                            if response:
                                # record and remove the dealing agents
                                logging.info(f"An announcement to {announcement_type} for ${price} was accepted by agent {responding_agent._id} at iteration {iteration}.")
                                responding_agent_id = responding_agent._id
                                responding_agent_reservation_price = responding_agent._reservation_price
                                for idx in sorted([i, j], reverse=True):
                                    del remaining_agents[idx]
                                transaction_made = True
                                break

                    if transaction_made:
                        announcing_agent.update_own_announcement_history(price, round, iteration, accepted=True)
                        break
                    else:
                        announcing_agent.update_own_announcement_history(price, round, iteration, accepted=False)
                    # Maybe it would be cleaner to use a while loop in this case, rather than shuffling
                else:
                    logging.error("The price announcement from the LLM could not be parsed.")

            
            if not announcement_made:
                logging.info(f'No announcement was made at iteration {iteration}.')
            
            # update the prompt with history of what happened at this iteration
            if announcement_made:
                if transaction_made:
                    history += f"In round {round} at iteration {iteration}, an announcement to {announcement_type} for ${price} was accepted.\n"
                else:
                    history += f"In round {round} at iteration {iteration}, an announcement to {announcement_type} for ${price} was made but no one responded.\n"
            else:
                history += f"In round {round} at iteration {iteration}, no announcement was made.\n"

            # store the data from the current iteration
            # TODO: this needs to be moved inside the solitcation loop because now we are only storing the last announcement  
            data_iterations.append({
                'round': round, 'iteration': iteration, 'price': price,
                'announcement': announcement_made, 
                'transaction': transaction_made,
                'announcement_type': announcement_type,
                'announcing_agent_id': announcing_agent_id,
                'announcing_agent_reservation_price': announcing_agent_reservation_price,
                'responding_agent_id': responding_agent_id,
                'responding_agent_reservation_price': responding_agent_reservation_price
            })

    # save the results to CSV
    output_filename = Path(__file__).parent.resolve() / f"results/experiment_{EXPERIMENT_ID}.csv"
    pd.DataFrame.from_dict(data_iterations).to_csv(output_filename)
    agents_dfs = []
    for agent in agents:
        df_data_agent = pd.DataFrame.from_dict(agent.own_history_data)
        df_data_agent['id'] = agent._id
        df_data_agent['reservation_price'] = agent._reservation_price
        df_data_agent['type'] = type(agent).__name__
        agents_dfs.append(df_data_agent)
    df_data_agents = pd.concat([df for df in agents_dfs]).reset_index(drop=True)
    output_filename = Path(__file__).parent.resolve() / f"results/agent_histories/experiment_{EXPERIMENT_ID}.csv"
    df_data_agents.to_csv(output_filename)


    #TODO: add an arg parser to be able to run the experiments as script once finished

    logging.info("All done.")
    

if __name__ == "__main__":
    main()
