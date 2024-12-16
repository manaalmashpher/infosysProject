from groq import Groq

apikey="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"
client = Groq(api_key=apikey)

def exercise_planner(user_input):
    prompt = f"""
        You are a helpful Gym Coach and experienced exercise planner
        This is the body part I want to train :{user_input}
        
        Your response should be max of 6 exercises
        with 10-12 repetition of each exercise
        Just exercise and it's repetitons

        Example
        # body part: Biceps
        # results:
        # Bicep Curls x 10

        
"""
    results = client.chat.completions.create(
        messages=[
            {
                "role":"user",
                "content":prompt,
            }
        ],
        model="llama-3.3-70b-versatile",
    )

    response = results.choices[0].message.content
    print(response)


if __name__ == "__main__":
    user_input = input("Enter which body part you want to train: ")
    exercise_planner(user_input)