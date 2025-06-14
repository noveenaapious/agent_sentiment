import sys
from crew import Sentiment

def run():
    """
    Run the crew.
    """
    inputs ={"text": "Im feeling depressed and having a lack of sleep",
            "user":"Fiana",
             "days":"12 days"
            }

    Sentiment().crew().kickoff(inputs=inputs)
run()




