from openai import OpenAI
from abc import ABC, abstractmethod
from loguru import logger
import os



class SystemBase(ABC):
    def __init__(self, name, max_retries=2, verbose=True):
        self.name = name
        self.max_retries = max_retries
        self.verbose = verbose

    @abstractmethod
    def execute(self, *args, **kwargs):
        pass

    def call_model(self, messages, model, base_url, api_key, temperature=0.7, max_tokens=150):
        """
        Calls the model via OpenAI Endpoint Call and retrieves the response.

        Args:
            messages (list): A list of message dictionaries.
            model (str): Model to use
            base_url (str): Base URL to act with
            api_key (str): Your API KEY 
            temperature (float): Sampling temperature.
            max_tokens (int): Maximum number of tokens in the response.

        Returns:
            str: The content of the model's response.
        """
        retries = 0
        while retries < self.max_retries:
            try:
                if self.verbose:
                    logger.info(f"[{self.name}] Sending messages to Endpoint:")
                    for msg in messages:
                        logger.debug(f"  {msg['role']}: {msg['content']}")

                # Call the Endpoint
                client = OpenAI(base_url=base_url,api_key=api_key)
                response = client.chat.completions.create(
                model= model,
                messages= messages,
                )                

                # Parse the response to extract the text content
                reply = response.choices[0].message.content 
                
                if self.verbose:
                    logger.info(f"[{self.name}] Received response: {reply}")
                
                return reply
            except Exception as e:
                retries += 1
                logger.error(f"[{self.name}] Error during Endpoint call: {e}. Retry {retries}/{self.max_retries}")
        raise Exception(f"[{self.name}] Failed to get response from Endpoint after {self.max_retries} retries.")
