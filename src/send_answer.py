import os
import requests
from dotenv import load_dotenv, find_dotenv


def send_answer(answer, challenge):
    '''Send answers to the code nation challenge through the corresponding URL'''
    # find .env automatically by walking up directories until it's found
    dotenv_path = find_dotenv()

    # load up the entries as environment variables
    load_dotenv(dotenv_path)

    token = os.environ.get("TOKEN")
    email = os.environ.get("EMAIL")

    dict_answer = {
      "token": token,
      "email": email,
      "answer": answer.to_dict('records')
    }

    url = {
        1: 'https://api.codenation.com.br/v1/user/acceleration/data-science/challenge/enem-1/submit',
        2: 'https://api.codenation.com.br/v1/user/acceleration/data-science/challenge/enem-2/submit',
        3: 'https://api.codenation.com.br/v1/user/acceleration/data-science/challenge/enem-3/submit',
    }

    response = requests.post(url[challenge], json=dict_answer)
    return response
