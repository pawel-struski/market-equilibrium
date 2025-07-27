from abc import ABC, abstractmethod
from typing import Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import re
import os

from llm_setup import act_gpt4_test

# NOTE: the bid/offer terminology is very specific to finance, but maybe that is good

# experiment parameters
N_ROUNDS = 5
N_ITER = 5


def extract_price(text: str) -> float:
    """
    Extracts a price from text if available and returns it as float.
    """
    match = re.search(r'(?:\$)?\s*(\d{1,3}(?:,\d{3})*|\d+)(\.\d{2})?', text)
    if match:
        number_str = match.group(1) + (match.group(2) if match.group(2) else '')
        number_str = number_str.replace(',', '')  # Remove commas
        return float(number_str)
    return None


def extract_response(text: str) -> bool:
    """
    Extracts a boolean indicator of whether a deal is accepted from a textual response.
    """
    if "yes" in text.lower():
        return True
    else:
        return False

# cleaner design: implement respond/announce method in the parent class
# and let the child classes only differ in the prompt generation

class Agent(ABC):
    def __init__(self, id: int, reservation_price: float):
        self._reservation_price = reservation_price
        self._id = id

    @abstractmethod
    def respond(self, price: float, history: str, round: int, iteration: int) -> bool: ...

    @abstractmethod
    def announce(self, history: str, round: int, iteration: int) -> float: ...


class Buyer(Agent):

    instructions = f"""
    You are a buyer. Your task is to buy a unit of good at the lowest possible price, 
    but no higher than your reservation price. There will be {N_ROUNDS} rounds.
    Your reservation price will be revealed to you, and only you, at the start.
    Under no condition can you buy above your reservation price. 
    However, buying at a price equal to your reservation price is perfectly acceptable and preferred than not buying at all.
    
    There are 11 sellers and 11 buyers (including you) in the room.
    Any buyer or seller is free at any time to raise his hand and make a verbal offer to buy/sell.
    Any buyer or seller is free to accept anb in th offer, in whicand h case a binding contract has been formed, the transaction occurs and the buyer and seller drop out of the market (no longer permitted to do anything for the remainder of that ronud).

    Each round, you want to buy an additional unit of the good and are able to 
    participate in the market irrespective of whether you transacted during the previous day.
    Each round, a maximum of {N_ITER} transactions can be made.
    \n
    """    

    def __init__(self, id: int, reservation_price: float):
        super().__init__(id, reservation_price)

    def respond(self, price: float, history: str, round: int, iteration: int) -> bool:
        prompt = self.instructions + f"Your reservation price is {self._reservation_price}.\n" + history + f"This is round {round}/{N_ROUNDS} iteration {iteration}/{N_ITER}. Someone is offering to sell at ${price:.2f}. Do you buy? Only answer with a yes or no."
        # call the LLM with the prompt and get the response
        print(f"Buyer with id {self._id} calling the LLM with the prompt: \n{prompt}")
        response_text = act_gpt4_test(prompt)
        print(f"LLM response: {response_text}")
        response = extract_response(response_text)
        return response
        
    def announce(self, history: str, round: int, iteration: int) -> float:
        prompt = self.instructions + f"Your reservation price is {self._reservation_price}.\n" + history + f"This is round {round}/{N_ROUNDS} iteration {iteration}/{N_ITER}. Do you want to announce a bid to buy? If so, what is your bid price? Answer only with a number."
        # call the LLM with the prompt and get the response
        print(prompt)
        print(f"Buyer with id {self._id} calling the LLM with the prompt: \n{prompt}")
        announcement_text = act_gpt4_test(prompt)
        print(f"LLM reponse: {announcement_text}")
        price = extract_price(announcement_text)
        return price
    

class Seller(Agent):

    instructions = f"""
    You are a seller. Your task is to sell a unit of your good at the highest possible price, 
    but no lower than your reservation price. There will be {N_ROUNDS} rounds.
    Your reservation price will be revealed to you, and only you, at the start.
    Under no condition can you sell below your reservation price. 
    However, selling at a price equal to your reservation price is perfectly acceptable and preferred than not selling at all.
    
    There are 11 buyers and 11 sellers (including you) in the room.
    Any buyer or seller is free at any time to raise his hand and make a verbal offer to buy/sell.
    Any buyer or seller is free to accept an offer, in whicand h case a binding contract has been formed, the transaction occurs and the buyer and seller drop out of the market (no longer permitted to do anything for the remainder of that round).

    Each round, you receive an additional unit of the good and are able to 
    participate in the market irrespective of whether you transacted during the previous day.
    Each round, a maximum of {N_ITER} transactions can be made. You can only make one transaction.
    \n
    """

    def __init__(self, id: int, reservation_price: float):
        super().__init__(id, reservation_price)

    def respond(self, price: float, history: str, round: int, iteration: int) -> bool:
        prompt = self.instructions + f"Your reservation price is {self._reservation_price}.\n" + history + f"This is round {round}/{N_ROUNDS} iteration {iteration}/{N_ITER}. Someone is offering to buy at ${price:.2f}. Do you sell? Only answer with a yes or no."
        # call the LLM with the prompt and get the response
        print(f"Seller with id {self._id} calling the LLM with the prompt: \n{prompt}")
        response_text = act_gpt4_test(prompt)
        print(f"LLM response: {response_text}")
        response = extract_response(response_text)
        return response
    
    def announce(self, history: str, round: int, iteration: int) -> float:
        prompt = self.instructions + f"Your reservation price is {self._reservation_price}.\n" + history + f"This is round {round}/{N_ROUNDS} iteration {iteration}/{N_ITER}. Do you want to announce an offer to sell? If so, what is your asking price? Answer only with a number."
        # call the LLM with the prompt and extract the response price
        print(f"Seller with id {self._id} calling the LLM with the prompt: \n{prompt}")
        announcement_text = act_gpt4_test(prompt)
        print(f"LLM reponse: {announcement_text}")
        price = extract_price(announcement_text)
        return price

# TODO: need to inject the round/iteration info to the prompt

def main():

    # generate reservation prices for the buyers and sellers
    # symmetric for now
    buyers_reservation_prices = np.round(np.linspace(0.8, 3.2, 11), 2)
    sellers_reservation_prices = np.round(np.linspace(0.8, 3.2, 11), 2)
    print(f"Buyers reservation prices: {buyers_reservation_prices}")
    print(f"Sellers reservation prices: {sellers_reservation_prices}")

    # initialise the agents (buyers and sellers with symmetric res prices)
    agents = []
    for id, res_price in enumerate(buyers_reservation_prices):
        agents.append(Buyer(id, res_price))
    for id, res_price in enumerate(buyers_reservation_prices):
        agents.append(Seller(id, res_price))

    # intitialise a dataframe to record the results
    df_data = pd.DataFrame(columns=['round', 'iteration', 'price', 'announcement', 
                                    'transaction', 'announcement_type'])

    # conduct each round of the expeirment seqeuentially
    history = ""
    for round in range(1, N_ROUNDS+1):
        remaining_agents = agents.copy()

        # solicit announcements from all agents at each iteration 
        for iteration in range(1, N_ITER+1):
            # reset the iteration info
            transaction_made = False
            announcement_made = False
            announcement_type = ""
            
            # shuffle the agents order
            np.random.shuffle(remaining_agents)

            # solicit a price announcement
            print("Prompting agents for an announcement...")
            for i, agent in enumerate(remaining_agents):
                price = agent.announce(history, round, iteration)
                if price is not None:
                    announcement_made = True
                    if isinstance(agent, Seller):
                        announcement_type = "sell"
                    else:
                        announcement_type = "buy"
                    print(f"An announcement to {announcement_type} for ${price} was made by agent {agent._id} at iteration {iteration}.")
                    #TODO: what if no announcement made after going through all?

                    # obtain a response to the announcement
                    print("Prompting agents for a response to the announcement...")
                    for j, agent in enumerate(remaining_agents):
                        if (isinstance(agent, Seller) and announcement_type == "buy") or (isinstance(agent, Buyer) and announcement_type == "sell"):
                            response = agent.respond(price, history, round, iteration)
                            if response:
                                # record and remove the dealing agents
                                print(f"An announcement to {announcement_type} for ${price} was accepted by agent {agent._id} at iteration {iteration}.")
                                for idx in sorted([i, j], reverse=True):
                                    del remaining_agents[idx]
                                transaction_made = True
                                break
            
            if not announcement_made:
                print(f'No announcement was made at iteration {iteration}.')
            
            # update the prompt with history of what happened at this iteration
            if announcement_made:
                if transaction_made:
                    history += f"In round {round} at iteration {iteration}, an announcement to {announcement_type} for ${price} was accepted.\n"
                else:
                    history += f"In round {round} at iteration {iteration}, an announcement to {announcement_type} for ${price} was made but no one responded.\n"
            else:
                history += f"In round {round} at iteration {iteration}, no announcement was made.\n"

            # store the data from the current iteration
            data_iter = {'round': [round], 'iteration': [iteration], 'price': [price],
                         'announcement': [announcement_made], 
                         'transaction': [transaction_made],
                         'announcement_type': [announcement_type]}
            df_data = pd.concat([df_data, pd.DataFrame(data=data_iter)], 
                                ignore_index=True)

    # save the results to CSV
    output_filename = Path(__file__).parent.resolve() / "results/experiment_3.csv"
    df_data.to_csv(output_filename)
    #TODO: you can record how much profit the participants made and how traded or not
    #TODO: need to check if the agents are not bidding above or asking below their reservation price
    #TODO: add an arg parser to be able to run the experiments as script once finished

    print("All done.")
    print("Exit.")  
        


    
    # ask for bids/offers to all agents
    # choose one randomly
    # ask for a response to all the agents of the opposing type to the one announcing
    # if a deal is made:
        # record the deal data (choose one response randomly if many)
        # remove the two involved agents from the round
    # update the prompt recording exactly what happened during this iteration
    # continue for a fixed number of iterations or until no deals are made any more

    

if __name__ == "__main__":
    main()
