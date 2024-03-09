from openai import OpenAI
import requests
import sseclient
import json
import sys

import json
import requests
import sseclient  # Assuming you have sseclient installed
import sys
from colorama import init, Fore

init()

class OpenAIChatbot:
    """Interacts with the OpenAI API for chatbot functionality."""

    def __init__(self, api_url, character_data, model_data, api_key, save_history=True):
        """Initializes the OpenAIChatbot.

        Args:
            api_url: The base URL of the OpenAI API.
            character_data: Path to a JSON file containing character data.
            model_data: Path to a JSON file containing model configuration.
            api_key: Your OpenAI API key.
            save_history: If True, saves chat history.
        """
        self._api_key = api_key
        self._url = api_url
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        self._save_history = save_history
        self._history = []
        self._chat_chain = []

        with open(character_data, 'r') as file:
            self._character_data = json.load(file)
        with open(model_data, 'r') as file:
            self._model_data = json.load(file)

        self._original_context = self._character_data["context"]

    def generate_context(self, biography_context):
        """Generates a context message with biographical information.

        Args:
            biography_context: The biographical information to include.

        Returns:
            A dictionary representing the "system" role message.
        """
        full_context = (f"{self._original_context}, The following is some biographical " 
                        f"context. Use it for crafting your answer but do not respond "
                        f"out of character. INFO_STARTS: {biography_context} INFO_ENDS.")
        return {"role": "system", "content": full_context}

    def generate_message(self, query):
        """Generates a user message.

        Args:
            query: The user's query.

        Returns:
            A dictionary representing the "user" role message.
        """
        return {"role": "user", "content": query}

    def generate_chatbot_response(self, response):
        """Generates a chatbot response message.

        Args:
            response: The OpenAI API's generated response.

        Returns:
            A dictionary representing the "assistant" role message.
        """
        return {"role": "assistant", "content": response}

    def get_stream_response(self, chat_chain):
        """Gets a streaming response from the OpenAI API.

        Args:
            chat_chain: The current conversation history.

        Returns:
            A requests.Response object in streaming mode.
        """
        self._model_data["messages"] = chat_chain
        return requests.post(self._url,
                             json=self._model_data,
                             headers=self._headers, 
                             stream=True)

    def process_stream(self, stream_response):
        """Processes the streaming response from OpenAI.

        Args:
            stream_response: A requests.Response object in streaming mode.

        Returns:
            The full accumulated response content.
        """
        client = sseclient.SSEClient(stream_response)
        full_response_content = ""

        for event in client.events():
            if event.data.strip() != "[DONE]":
                try:
                    payload = json.loads(event.data)
                    delta_content = payload['choices'][0].get('delta', {}).get('content', '')
                    full_response_content += delta_content
                    print(delta_content, end='') 
                    sys.stdout.flush()
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")

        return full_response_content

    def drop_context(self):
        """Removes the existing 'system' context message from the chat history."""
        system_index = next((i for i, message in enumerate(self._chat_chain) 
                             if message["role"] == "system"), None)
        if system_index is not None:
            self._chat_chain.pop(system_index)

    def chat(self, query, context, print_context=False):
        """Handles a single chat interaction with the OpenAI API.

        Args:
            query: The user's query.
            context: Biographical context to provide to the chatbot.

        Returns:
            The chatbot's response as a string.
        """
        self.drop_context()

        new_context = self.generate_context(context)
        print("RAG prompt:\n" + Fore.GREEN + new_context["content"] + Fore.RESET)
        self._chat_chain.append(new_context)
        self._history.append(new_context)

        new_user_message = self.generate_message(query)
        self._chat_chain.append(new_user_message)
        self._history.append(new_user_message)

        stream = self.get_stream_response(self._chat_chain)
        response = self.process_stream(stream)

        new_chatbot_response = self.generate_chatbot_response(response)
        self._chat_chain.append(new_chatbot_response)
        self._history.append(new_chatbot_response)

        return response

    def get_history(self):
        """Retrieves the chat history.

        Returns:
            A list of chat messages (dictionaries).
        """
        return self._history 

        
        
        

