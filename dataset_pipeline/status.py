import json
from mammography_agent.datasets.manager import statuses
if __name__=="__main__": print(json.dumps(statuses(),indent=2))
