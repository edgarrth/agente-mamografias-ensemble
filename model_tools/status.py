import json
from mammography_agent.model_client import status
if __name__=="__main__": print(json.dumps(status(),indent=2))
