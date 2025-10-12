import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import logging
import shutil
import copy

from utils import load_config, get_experiment_commit_hash
from agent import (Agent, AgentType, AnnouncementType, AgentPromptConfig, 
                   AgentLLMConfig, GeneralPromptConfig, AgentPromptKeywords,
                   ExperimentConfig)


def main(config_name: str):

    # load the experiment config
    experiment_config_path = f"configs/{config_name}.yaml"
    exp_config_path = Path(__file__).parent.resolve() / experiment_config_path
    config = load_config(exp_config_path)

    # Determine experiment timestamp and the commit hash of the latest change
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    relevant_paths = [
        "experiment.py",
        "llm_setup.py",
        "agent.py",
        "configs/"
    ]
    commit_hash = get_experiment_commit_hash(relevant_paths)[:7]

    # Create a unique results folder
    outdir = Path(__file__).parent.resolve() / "results" / f"{config_name}_{commit_hash}_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=False)
    log_path = outdir / "sims" / "logs"
    log_path.mkdir(parents=True, exist_ok=False)
    data_path = outdir / "sims" / "data"
    data_path.mkdir(parents=True, exist_ok=False)

    # Save the config used
    shutil.copy(experiment_config_path, outdir / "config_used.yaml")
    
    # extract experiment-level config elements
    N_ROUNDS = config["experiment"]["n_rounds"]
    N_ITER = config["experiment"]["n_iter"]
    N_SIMS = config["experiment"]["n_sims"]
    experiment_config = ExperimentConfig(N_ROUNDS=N_ROUNDS, N_ITER=N_ITER)

    # extract LLM config elements
    llm_config = AgentLLMConfig(**config["agent"]["llm"])
    
    # extract prompt config elements
    prompt_config = config["agent"]["prompts"]

    # create a general prompt config instance (common to both buyers and sellers)
    general_prompt_config = GeneralPromptConfig(**prompt_config["general"])
       
    # extract buyer- and seller- specific prompt elements and create agent prompt configs
    buyer_prompt_config = AgentPromptConfig(
        general=general_prompt_config,
        main_keywords=AgentPromptKeywords(**prompt_config["buyer"]["main_keywords"]),
        response_prompt=prompt_config["buyer"]["response_prompt"],
        announcement_prompt=prompt_config["buyer"]["announcement_prompt"],
    )

    seller_prompt_config = AgentPromptConfig(
        general=general_prompt_config,
        main_keywords=AgentPromptKeywords(**prompt_config["seller"]["main_keywords"]),
        response_prompt=prompt_config["seller"]["response_prompt"],
        announcement_prompt=prompt_config["seller"]["announcement_prompt"],
    )
    
    # generate symmetric reservation prices for the buyers and sellers
    bp = config["experiment"]["buyers_reservation_prices"]
    sp = config["experiment"]["sellers_reservation_prices"]
    buyers_reservation_prices = np.round(np.linspace(bp["min"], bp["max"], 
                                                     bp["num"]), 2)
    sellers_reservation_prices = np.round(np.linspace(sp["min"], sp["max"], 
                                                     sp["num"]), 2)

    # repeat the experiment N_SIMS times
    print(f"Running the experiment based on configs/{config_name}.yaml...")
    for sim in range(1, N_SIMS+1):
        print(f"Running simulation {sim} out of {N_SIMS}...")

        # configure the logger for that simulation
        logger = logging.getLogger(f"sim{sim}")
        logger.setLevel(logging.INFO)
        # Prevent adding duplicate handlers if re-running in the same process
        if not logger.handlers:
            fh = logging.FileHandler(log_path/f"sim{sim}.log", mode='w')
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        logger.info(f"Buyers reservation prices: {buyers_reservation_prices}")
        logger.info(f"Sellers reservation prices: {sellers_reservation_prices}")

        # initialise agents
        agents = []
        for id, res_price in enumerate(buyers_reservation_prices):
            agents.append(Agent(
                id, res_price, AgentType.BUYER, buyer_prompt_config, llm_config,
                experiment_config, logger
            ))
        for id, res_price in enumerate(sellers_reservation_prices):
            agents.append(Agent(
                id, res_price, AgentType.SELLER, seller_prompt_config, llm_config,
                experiment_config, logger
            ))

        # conduct each round of the expeirment seqeuentially
        market_history = ""
        data_iterations = []
        for round in range(1, N_ROUNDS+1):
            remaining_agents = agents.copy()

            # solicit announcements from all agents at each iteration 
            for iteration in range(1, N_ITER+1):
                # reset the iteration info
                transaction_made = False
                announcement_made = False
                announcement_type = None
                responding_agent_id = None
                announcing_agent_id = None
                announcing_agent_reservation_price = None
                responding_agent_reservation_price = None
                price = None
                
                # shuffle the agents order
                np.random.shuffle(remaining_agents)

                # solicit a price announcement
                logger.info("Prompting agents for an announcement...")
                for i, announcing_agent in enumerate(remaining_agents):
                    price = announcing_agent.announce(market_history, round, iteration)
                    if price is not None:
                        announcement_made = True
                        if announcing_agent._type == AgentType.SELLER:
                            announcement_type = AnnouncementType.SELL
                        else:
                            announcement_type = AnnouncementType.BUY
                        logger.info(f"An announcement to {announcement_type.value} for ${price} was made by agent {announcing_agent._id} at iteration {iteration}.")
                        announcing_agent_id = announcing_agent._id
                        announcing_agent_reservation_price = announcing_agent._reservation_price
                        
                        # filter and shuffle potential respondents
                        potential_respondents = [
                            agent for agent in remaining_agents
                            if (agent._type == AgentType.SELLER and announcement_type == AnnouncementType.BUY) or 
                            (agent._type == AgentType.BUYER and announcement_type == AnnouncementType.SELL)
                        ]
                        np.random.shuffle(potential_respondents)

                        # obtain a response to the announcement
                        logger.info("Prompting agents for a response to the announcement...")
                        for j, responding_agent in enumerate(potential_respondents):
                            response = responding_agent.respond(price, market_history, round, iteration)
                            responding_agent.update_own_responding_history(price, round, iteration, accepted=response)
                            responding_agent_id = responding_agent._id
                            responding_agent_reservation_price = responding_agent._reservation_price
                            # Save every response, not just the last one
                            data_iterations.append({
                                'round': round,
                                'iteration': iteration,
                                'price': price,
                                'announcement': announcement_made,
                                'transaction': response,
                                'announcement_type': announcement_type.value,
                                'announcing_agent_id': announcing_agent_id,
                                'announcing_agent_reservation_price': announcing_agent_reservation_price,
                                'responding_agent_id': responding_agent_id,
                                'responding_agent_reservation_price': responding_agent_reservation_price
                            })
                            if response:
                                # record and remove the dealing agents
                                logger.info(f"An announcement to {announcement_type.value} for ${price} was accepted by agent {responding_agent._id} at iteration {iteration}.")
                                for idx in sorted([i, j], reverse=True):
                                    del remaining_agents[idx]
                                transaction_made = True
                                break

                        if transaction_made:
                            announcing_agent.update_own_announcement_history(price, round, iteration, accepted=True)
                            market_history += f"In round {round} at iteration {iteration}, an announcement to {announcement_type.value} for ${price} was accepted.\n"
                            break
                        else:
                            announcing_agent.update_own_announcement_history(price, round, iteration, accepted=False)
                            market_history += f"In round {round} at iteration {iteration}, an announcement to {announcement_type.value} for ${price} was made but no one responded.\n"
                        # Maybe it would be cleaner to use a while loop in this case, rather than shuffling
                    else:
                        logger.error("The price announcement from the LLM could not be parsed.")
    
                if not announcement_made:
                    logger.info(f'No announcement was made at iteration {iteration}.')
                    market_history += f"In round {round} at iteration {iteration}, no announcement was made.\n"
                
        # save the results to CSV
        output_filename = data_path / f"iteration_history_{sim}.csv"
        pd.DataFrame.from_dict(data_iterations).to_csv(output_filename)
        agents_dfs = []
        for agent in agents:
            df_data_agent = pd.DataFrame.from_dict(agent.own_history_data)
            df_data_agent['id'] = agent._id
            df_data_agent['reservation_price'] = agent._reservation_price
            df_data_agent['type'] = agent._type.value
            agents_dfs.append(df_data_agent)
        df_data_agents = pd.concat([df for df in agents_dfs]).reset_index(drop=True)
        agent_output_filename = data_path / f"agent_histories_{sim}.csv"
        df_data_agents.to_csv(agent_output_filename)

        logger.info("Simulation done.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a market equilibrium experiment.")
    parser.add_argument("config_name", type=str, help="Name of the yaml config file to use (e.g. exp1)")
    args = parser.parse_args()
    main(args.config_name)
