from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd
import numpy as np
import re

# NOTE: the bid/offer terminology is very specific to finance, but maybe that is good


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
    def respond(self, price: float) -> str:
        pass

    @abstractmethod
    def announce(self) -> str:
        pass


class Buyer(Agent):
    def __init__(self, id: int, reservation_price: float):
        super().__init__(id, reservation_price)

    def respond(self, price: float) -> str:
        # TODO: refine the prompt
        prompt = f"You are a Buyer. Your reservation price is {self._reservation_price}. Someone is offering to sell at ${price}. Do you buy?"
        # call the LLM with the prompt and get the response
        print(prompt)
        print("<Here the LLM will be called>")
        response_text = "<LLM responding>"
        response = extract_response(response_text)
        return response
        
    def announce(self) -> str:
        # TODO: refine the prompt
        prompt = f"You are a Buyer. Your reservation price is {self._reservation_price}. Do you want to announce a bid to buy? If so, what is your bid?"
        # call the LLM with the prompt and get the response
        print(prompt)
        print("<Here the LLM will be called>")
        announcement_text = "<LLM responding>"
        price = extract_price(announcement_text)
        return price
    

class Seller(Agent):
    def __init__(self, id: int, reservation_price: float):
        super().__init__(id, reservation_price)

    def respond(self, price: float) -> bool:
        # TODO: refine the prompt
        prompt = f"You are a Seller. Your reservation price is {self._reservation_price}. Someone is offering to buy at ${price}. Do you sell?"
        # call the LLM with the prompt and get the response
        print(prompt)
        print("<Here the LLM will converts into abe called>")
        response_text = "<LLM responding>"
        response = extract_response(response_text)
        return response
    
    def announce(self) -> str:
        # TODO: refine the prompt
        prompt = f"You are a Seller. Your reservation price is {self._reservation_price}. Do you want to announce an offer to sell? If so, what is your offer price?"
        # call the LLM with the prompt and get the response
        print(prompt)
        print("<Here the LLM will be called>")
        announcement_text = "<LLM responding>"
        price = extract_price(announcement_text)
        return price


def simulate_reservation_prices():
    # symmetric case for buyer's and sellers
    # the middle point is 2 
    reservation_prices = np.linspace(0.8, 3.2, 11)
    return reservation_prices


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
    df_data = pd.DataFrame(columns=['iteration', 'price', 'announcement', 
                                    'transaction', 'announcement_type'])

    # simulate a single market round
    n_iter = 2
    remaining_agents = agents.copy()
    round_history = ""
    for iteration in range(0, n_iter):
        # reset the history
        transaction_made = False
        announcement_made = False
        announcement_type = ""
        
        # shuffle the agents order
        np.random.shuffle(remaining_agents)

        # obtain a price announcement
        print("Prompting agents for an announcement...")
        for i, agent in enumerate(remaining_agents):
            price = agent.announce()
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
                        response = agent.respond(price)
                        if response:
                            # record and remove the dealing agents
                            print(f"An announcement to {announcement_type} for ${price} was accepted by agent {agent._id} at iteration {iteration}.")
                            remaining_agents.pop(i)
                            remaining_agents.pop(j)
                            transaction_made = True
                            break
        
        if not announcement_made:
            print(f'No announcement was made at iteration {iteration}.')
        
        # update the prompt with history of what happened at this iteration
        if announcement_made:
            if transaction_made:
                round_history += f"An announcement to {announcement_type} for ${price} was accepted at iteration {iteration}.\n"
            else:
                round_history += f"An announcement to {announcement_type} for ${price} was made at iteration {iteration} but noone responded.\n"
        else:
            round_history += f"No announcement was made at iteration {iteration}.\n"

        # store the data from the current iteration
        df_data.loc[iteration] = [iteration, price, announcement_made, 
                                  transaction_made, announcement_type]

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
