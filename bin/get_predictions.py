import json
import requests

sentence = '''
lets raid their subreddit and harass them in the comments
'''

model_name = 'cth'

batch = [sentence]

input_data = {"instances": batch}
print(json.dumps(input_data))
r = requests.post(
    f"http://localhost:8501/v1/models/{model_name}:predict",
    data=json.dumps(input_data)
)
print(r.json())